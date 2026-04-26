from __future__ import annotations

# A题：过滤设备监测 - 数据清洗与可视化分析脚本

# 运行位置：
#     在仓库根目录运行：
#         python ./03_goingon/北邮校赛/code/AnalyzeData.py

#     或进入 code 目录运行：
#         cd ./03_goingon/北邮校赛/code
#         python AnalyzeData.py

# 主要输出：
#     03_goingon/北邮校赛/plot/
#         01_device_curves/               10台设备透水率曲线，逐设备单独成图
#         02_maintenance_annotations/     10台设备维护点标注图，逐设备单独成图
#         03_season_boxplot.png           季节性箱线图
#         04_missing_and_outlier_summary.png
#         05_maintenance_gain_summary.png
#         06_natural_decline_rate.png
#         tables/                         CSV表格结果

# 说明：
#     1. 缺失透水率默认按“同一设备均值”填充；也可通过 --missing-strategy drop 删除。
#     2. 异常值默认只检查和记录，不直接替换，避免把维护后的真实跃升误判为异常并删掉。
#        如确需把 IQR 异常值也当作缺失值处理，可增加 --treat-outliers-as-missing。

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


# =========================
# 路径配置
# =========================
CODE_DIR = Path(__file__).resolve().parent
BASE_DIR = CODE_DIR.parent
RAW_DATA_DIR = BASE_DIR / "data" / "原始数据"
PLOT_DIR = BASE_DIR / "plot"
TABLE_DIR = PLOT_DIR / "tables"
DEVICE_CURVE_DIR = PLOT_DIR / "01_device_curves"
MAINTAIN_PLOT_DIR = PLOT_DIR / "02_maintenance_annotations"


# =========================
# 基础工具
# =========================
@dataclass
class DeviceData:
    device_code: str      # A1, A2, ..., A10
    sheet_name: str       # A_1, A_2, ..., A_10
    dataframe: pd.DataFrame


def configure_chinese_font() -> None:
    """尽量选择本机可用中文字体，避免图中中文乱码。"""
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Microsoft JhengHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            break
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"

    plt.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore", message=r"Glyph .* missing from font\(s\)")


def ensure_output_dirs() -> None:
    for directory in [PLOT_DIR, TABLE_DIR, DEVICE_CURVE_DIR, MAINTAIN_PLOT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def normalize_device_code(value: object) -> str:
    """统一设备编号格式：A_1 / a1 / A01 -> A1。"""
    text = str(value).strip().upper().replace("_", "")
    if text.startswith("A"):
        suffix = text[1:]
        if suffix.isdigit():
            return f"A{int(suffix)}"
    return text


def season_label(month: int) -> str:
    if month in (3, 4, 5):
        return "春"
    if month in (6, 7, 8):
        return "夏"
    if month in (9, 10, 11):
        return "秋"
    return "冬"


def add_common_time_axis(ax: plt.Axes) -> None:
    """两年左右的时间轴，每2个月一个主刻度，避免横轴过密。"""
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax.tick_params(axis="x", rotation=30)


def save_table_as_png(df: pd.DataFrame, title: str, output_path: Path, max_rows: int | None = None) -> None:
    """把较小的汇总表保存为PNG，便于直接放论文。"""
    show_df = df.copy()
    if max_rows is not None and len(show_df) > max_rows:
        show_df = show_df.head(max_rows)

    fig_height = max(2.8, min(16, 0.42 * len(show_df) + 1.6))
    fig_width = max(9, min(18, 1.4 * len(show_df.columns) + 2))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=10)

    table = ax.table(
        cellText=show_df.astype(str).values,
        colLabels=show_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================
# 数据读取
# =========================
def load_device_data() -> list[DeviceData]:
    """
    优先读取 DataProcess.py 已经生成的 CSV：
        附件1__A_1.csv ... 附件1__A_10.csv

    若 CSV 不存在，则直接读取：
        附件1.xlsx
    """
    devices: list[DeviceData] = []

    csv_files = sorted(RAW_DATA_DIR.glob("附件1__A_*.csv"))
    if csv_files:
        for csv_path in csv_files:
            sheet_name = csv_path.stem.split("__")[-1]
            device_code = normalize_device_code(sheet_name)
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            devices.append(_standardize_device_df(df, sheet_name, device_code))
        return sorted(devices, key=lambda x: int(x.device_code[1:]))

    excel_path = RAW_DATA_DIR / "附件1.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError(
            f"未找到附件1数据。请确认存在 {excel_path} 或附件1__A_*.csv"
        )

    xls = pd.ExcelFile(excel_path)
    for sheet_name in xls.sheet_names:
        if not sheet_name.upper().startswith("A"):
            continue
        device_code = normalize_device_code(sheet_name)
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        devices.append(_standardize_device_df(df, sheet_name, device_code))

    if not devices:
        raise ValueError("附件1中未找到 A_1 到 A_10 这类设备工作表。")

    return sorted(devices, key=lambda x: int(x.device_code[1:]))


def _standardize_device_df(df: pd.DataFrame, sheet_name: str, device_code: str) -> DeviceData:
    if "time" not in df.columns or "per" not in df.columns:
        raise ValueError(f"{sheet_name} 缺少 time 或 per 列，当前列：{df.columns.tolist()}")

    clean_df = df[["time", "per"]].copy()
    clean_df = clean_df.rename(columns={"time": "时间", "per": "透水率"})
    clean_df["时间"] = pd.to_datetime(clean_df["时间"], errors="coerce")
    clean_df["透水率"] = pd.to_numeric(clean_df["透水率"], errors="coerce")
    clean_df = clean_df.sort_values("时间").reset_index(drop=True)

    return DeviceData(device_code=device_code, sheet_name=sheet_name, dataframe=clean_df)


def load_maintenance_data() -> pd.DataFrame:
    """
    优先读取 DataProcess.py 已经生成的 附件2.csv；
    若不存在，则直接读取 附件2.xlsx。
    """
    csv_path = RAW_DATA_DIR / "附件2.csv"
    excel_path = RAW_DATA_DIR / "附件2.xlsx"

    if csv_path.exists():
        maintenance = pd.read_csv(csv_path, encoding="utf-8-sig")
    elif excel_path.exists():
        maintenance = pd.read_excel(excel_path)
    else:
        raise FileNotFoundError(f"未找到维护记录文件：{csv_path} 或 {excel_path}")

    required_cols = {"编号", "日期", "维护类型"}
    if not required_cols.issubset(set(maintenance.columns)):
        raise ValueError(f"附件2列名异常，当前列：{maintenance.columns.tolist()}")

    maintenance = maintenance[["编号", "日期", "维护类型"]].copy()
    maintenance["编号"] = maintenance["编号"].map(normalize_device_code)
    maintenance["日期"] = pd.to_datetime(maintenance["日期"], errors="coerce")
    maintenance["维护类型"] = maintenance["维护类型"].astype(str).str.strip()
    maintenance = maintenance.dropna(subset=["日期"])
    maintenance = maintenance.sort_values(["编号", "日期"]).reset_index(drop=True)

    return maintenance


# =========================
# 数据清洗与检查
# =========================
def inspect_and_clean_devices(
    devices: list[DeviceData],
    missing_strategy: str,
    treat_outliers_as_missing: bool,
) -> tuple[list[DeviceData], pd.DataFrame, pd.DataFrame]:
    """
    输出：
        cleaned_devices: 清洗后数据
        quality_summary: 缺失、重复、异常值汇总
        outlier_detail: IQR异常值明细
    """
    cleaned_devices: list[DeviceData] = []
    summary_rows: list[dict[str, object]] = []
    outlier_rows: list[dict[str, object]] = []

    for device in devices:
        df = device.dataframe.copy()
        original_rows = len(df)

        invalid_time_count = int(df["时间"].isna().sum())
        missing_per_count = int(df["透水率"].isna().sum())
        duplicated_time_count = int(df["时间"].duplicated().sum())

        valid_per = df["透水率"].dropna()
        q1 = valid_per.quantile(0.25)
        q3 = valid_per.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_mask = (df["透水率"] < lower) | (df["透水率"] > upper)
        outlier_count = int(outlier_mask.sum())

        if outlier_count:
            outlier_part = df.loc[outlier_mask, ["时间", "透水率"]].copy()
            outlier_part["设备编号"] = device.device_code
            outlier_part["IQR下界"] = lower
            outlier_part["IQR上界"] = upper
            outlier_rows.extend(outlier_part[["设备编号", "时间", "透水率", "IQR下界", "IQR上界"]].to_dict("records"))

        # 时间异常无法用于时间序列，必须删除；重复时间保留同一时刻均值。
        df = df.dropna(subset=["时间"])
        df = df.groupby("时间", as_index=False)["透水率"].mean().sort_values("时间")

        if treat_outliers_as_missing:
            outlier_mask_after_group = (df["透水率"] < lower) | (df["透水率"] > upper)
            df.loc[outlier_mask_after_group, "透水率"] = np.nan

        if missing_strategy == "mean":
            fill_value = df["透水率"].mean()
            df["透水率"] = df["透水率"].fillna(fill_value)
            missing_action = f"按设备均值填充，均值={fill_value:.4f}"
        elif missing_strategy == "drop":
            df = df.dropna(subset=["透水率"])
            missing_action = "删除缺失透水率记录"
        else:
            raise ValueError(f"未知缺失值处理方式：{missing_strategy}")

        df = df.reset_index(drop=True)
        missing_after = int(df["透水率"].isna().sum())

        summary_rows.append(
            {
                "设备编号": device.device_code,
                "原始行数": original_rows,
                "无效时间数": invalid_time_count,
                "透水率缺失数": missing_per_count,
                "重复时间数": duplicated_time_count,
                "IQR异常值数": outlier_count,
                "IQR下界": round(float(lower), 4),
                "IQR上界": round(float(upper), 4),
                "异常值是否参与填充/删除": "是" if treat_outliers_as_missing else "否，仅记录",
                "缺失值处理": missing_action,
                "清洗后缺失数": missing_after,
                "清洗后行数": len(df),
            }
        )

        cleaned_devices.append(DeviceData(device.device_code, device.sheet_name, df))

    quality_summary = pd.DataFrame(summary_rows).sort_values("设备编号")
    outlier_detail = pd.DataFrame(outlier_rows)
    if not outlier_detail.empty:
        outlier_detail = outlier_detail.sort_values(["设备编号", "时间"]).reset_index(drop=True)

    return cleaned_devices, quality_summary, outlier_detail


def build_all_df(devices: list[DeviceData]) -> pd.DataFrame:
    frames = []
    for device in devices:
        df = device.dataframe.copy()
        df["设备编号"] = device.device_code
        df["月份"] = df["时间"].dt.month
        df["季节"] = df["月份"].map(season_label)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# =========================
# 画图
# =========================
def plot_each_device_curve(all_df: pd.DataFrame) -> None:
    for device_code, df in all_df.groupby("设备编号"):
        df = df.sort_values("时间")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["时间"], df["透水率"], linewidth=0.8)

        ax.set_title(f"{device_code} 透水率时间序列")
        ax.set_xlabel("时间")
        ax.set_ylabel("透水率")
        ax.grid(True, linestyle="--", alpha=0.35)
        add_common_time_axis(ax)

        fig.tight_layout()
        fig.savefig(DEVICE_CURVE_DIR / f"{device_code}_透水率曲线.png", dpi=300)
        plt.close(fig)


def _nearest_permeability(device_df: pd.DataFrame, dates: pd.Series, tolerance_days: int = 3) -> pd.DataFrame:
    """
    对维护日期找最近的透水率观测值，用于散点标注。
    """
    left = pd.DataFrame({"日期": pd.to_datetime(dates)}).sort_values("日期")
    right = device_df[["时间", "透水率"]].sort_values("时间")

    matched = pd.merge_asof(
        left,
        right,
        left_on="日期",
        right_on="时间",
        direction="nearest",
        tolerance=pd.Timedelta(days=tolerance_days),
    )
    return matched


def plot_each_device_with_maintenance(all_df: pd.DataFrame, maintenance: pd.DataFrame) -> None:
    color_map = {
        "中维护": "tab:orange",
        "大维护": "tab:red",
    }
    marker_map = {
        "中维护": "o",
        "大维护": "^",
    }

    for device_code, device_df in all_df.groupby("设备编号"):
        device_df = device_df.sort_values("时间")
        maintain_df = maintenance[maintenance["编号"] == device_code].copy()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(device_df["时间"], device_df["透水率"], linewidth=0.8, label="透水率")

        for maintain_type, group in maintain_df.groupby("维护类型"):
            matched = _nearest_permeability(device_df, group["日期"])
            valid = matched.dropna(subset=["透水率"])
            if valid.empty:
                continue

            ax.scatter(
                valid["日期"],
                valid["透水率"],
                s=42,
                color=color_map.get(maintain_type, "tab:green"),
                marker=marker_map.get(maintain_type, "s"),
                edgecolors="black",
                linewidths=0.4,
                label=maintain_type,
                zorder=5,
            )

            # 维护日竖线淡化显示，不抢曲线主体。
            for date in group["日期"]:
                ax.axvline(
                    date,
                    color=color_map.get(maintain_type, "gray"),
                    linestyle="--",
                    linewidth=0.5,
                    alpha=0.25,
                )

        ax.set_title(f"{device_code} 透水率曲线与维护点标注")
        ax.set_xlabel("时间")
        ax.set_ylabel("透水率")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best", fontsize=9)
        add_common_time_axis(ax)

        fig.tight_layout()
        fig.savefig(MAINTAIN_PLOT_DIR / f"{device_code}_维护点标注图.png", dpi=300)
        plt.close(fig)


def plot_season_boxplot(all_df: pd.DataFrame) -> None:
    season_order = ["春", "夏", "秋", "冬"]
    data = [
        all_df.loc[all_df["季节"] == season, "透水率"].dropna().values
        for season in season_order
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.boxplot(
        data,
        showmeans=True,
        patch_artist=True,
    )
    ax.set_xticks(range(1, len(season_order) + 1))
    ax.set_xticklabels(season_order)

    ax.set_title("10台设备透水率季节性箱线图")
    ax.set_xlabel("季节")
    ax.set_ylabel("透水率")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    season_stats = (
        all_df.groupby("季节")["透水率"]
        .agg(["count", "mean", "median"])
        .reindex(season_order)
        .round(3)
    )
    note = "；".join(
        f"{season}: n={int(row['count'])}, 均值={row['mean']:.2f}"
        for season, row in season_stats.iterrows()
        if not pd.isna(row["count"])
    )
    ax.text(
        0.5,
        -0.22,
        note,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_season_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    season_stats.to_csv(TABLE_DIR / "季节性统计表.csv", encoding="utf-8-sig")


# =========================
# 表格计算
# =========================
def calc_maintenance_gain_table(
    all_df: pd.DataFrame,
    maintenance: pd.DataFrame,
    before_window_days: int = 2,
    after_window_days: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    维护前后提升量：
        维护前：维护日期之前 before_window_days 天内最后一个观测值
        维护后：维护日期之后 after_window_days 天内第一个观测值
    """
    rows: list[dict[str, object]] = []

    for _, m in maintenance.iterrows():
        device_code = m["编号"]
        date = m["日期"]
        m_type = m["维护类型"]

        df = all_df[all_df["设备编号"] == device_code].sort_values("时间")
        before = df[(df["时间"] < date) & (df["时间"] >= date - pd.Timedelta(days=before_window_days))].tail(1)
        after = df[(df["时间"] >= date) & (df["时间"] <= date + pd.Timedelta(days=after_window_days))].head(1)

        if before.empty or after.empty:
            rows.append(
                {
                    "设备编号": device_code,
                    "维护日期": date.date(),
                    "维护类型": m_type,
                    "维护前时间": None,
                    "维护前透水率": np.nan,
                    "维护后时间": None,
                    "维护后透水率": np.nan,
                    "提升量": np.nan,
                    "备注": "维护前或维护后窗口内无观测值",
                }
            )
            continue

        before_time = before["时间"].iloc[0]
        after_time = after["时间"].iloc[0]
        before_per = float(before["透水率"].iloc[0])
        after_per = float(after["透水率"].iloc[0])

        rows.append(
            {
                "设备编号": device_code,
                "维护日期": date.date(),
                "维护类型": m_type,
                "维护前时间": before_time,
                "维护前透水率": round(before_per, 4),
                "维护后时间": after_time,
                "维护后透水率": round(after_per, 4),
                "提升量": round(after_per - before_per, 4),
                "备注": "",
            }
        )

    detail = pd.DataFrame(rows)
    summary = (
        detail.dropna(subset=["提升量"])
        .groupby("维护类型")["提升量"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "有效样本数",
                "mean": "平均提升量",
                "median": "中位提升量",
                "std": "标准差",
                "min": "最小提升量",
                "max": "最大提升量",
            }
        )
        .round(4)
    )

    return detail, summary


def calc_natural_decline_rate(
    all_df: pd.DataFrame,
    maintenance: pd.DataFrame,
    exclude_window_days: int = 3,
) -> pd.DataFrame:
    """
    自然下降速率：
        排除每次中/大维护前后若干天，避免维护冲击影响自然退化斜率；
        对剩余点做一元线性拟合：透水率 = intercept + slope * 天数。
    """
    rows: list[dict[str, object]] = []

    for device_code, df in all_df.groupby("设备编号"):
        df = df.sort_values("时间").copy()
        maintain_dates = maintenance.loc[maintenance["编号"] == device_code, "日期"]

        natural_mask = pd.Series(True, index=df.index)
        for date in maintain_dates:
            natural_mask &= ~df["时间"].between(
                date - pd.Timedelta(days=exclude_window_days),
                date + pd.Timedelta(days=exclude_window_days),
            )

        natural_df = df.loc[natural_mask].dropna(subset=["时间", "透水率"]).copy()
        if len(natural_df) < 2:
            rows.append(
                {
                    "设备编号": device_code,
                    "自然下降速率(每天)": np.nan,
                    "自然下降速率(每30天)": np.nan,
                    "自然下降速率(每年)": np.nan,
                    "拟合R2": np.nan,
                    "用于拟合样本数": len(natural_df),
                    "备注": "有效样本不足",
                }
            )
            continue

        x = (natural_df["时间"] - natural_df["时间"].min()).dt.total_seconds().to_numpy() / 86400.0
        y = natural_df["透水率"].to_numpy()

        slope, intercept = np.polyfit(x, y, deg=1)
        y_pred = slope * x + intercept
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        rows.append(
            {
                "设备编号": device_code,
                "自然下降速率(每天)": round(float(slope), 6),
                "自然下降速率(每30天)": round(float(slope * 30), 6),
                "自然下降速率(每年)": round(float(slope * 365), 6),
                "拟合R2": round(float(r2), 4),
                "用于拟合样本数": int(len(natural_df)),
                "备注": f"已排除维护日前后{exclude_window_days}天",
            }
        )

    return pd.DataFrame(rows).sort_values("设备编号")


# =========================
# 主流程
# =========================
def run_pipeline(
    missing_strategy: str,
    treat_outliers_as_missing: bool,
    before_window_days: int,
    after_window_days: int,
    exclude_window_days: int,
) -> None:
    configure_chinese_font()
    ensure_output_dirs()

    devices = load_device_data()
    maintenance = load_maintenance_data()

    cleaned_devices, quality_summary, outlier_detail = inspect_and_clean_devices(
        devices=devices,
        missing_strategy=missing_strategy,
        treat_outliers_as_missing=treat_outliers_as_missing,
    )
    all_df = build_all_df(cleaned_devices)

    # 1. 异常值与缺失值检查
    quality_summary.to_csv(TABLE_DIR / "异常值与缺失值处理汇总.csv", index=False, encoding="utf-8-sig")
    save_table_as_png(
        quality_summary,
        "异常值检查与缺失值处理汇总",
        PLOT_DIR / "04_missing_and_outlier_summary.png",
    )

    if outlier_detail.empty:
        pd.DataFrame(columns=["设备编号", "时间", "透水率", "IQR下界", "IQR上界"]).to_csv(
            TABLE_DIR / "IQR异常值明细.csv", index=False, encoding="utf-8-sig"
        )
    else:
        outlier_detail.to_csv(TABLE_DIR / "IQR异常值明细.csv", index=False, encoding="utf-8-sig")

    # 清洗后长表，后续建模可直接复用
    all_df.to_csv(TABLE_DIR / "清洗后透水率长表.csv", index=False, encoding="utf-8-sig")

    # 2. 10台设备透水率曲线：逐设备单独成图
    plot_each_device_curve(all_df)

    # 3. 维护点标注图：逐设备单独成图
    plot_each_device_with_maintenance(all_df, maintenance)

    # 4. 维护前后提升量表
    gain_detail, gain_summary = calc_maintenance_gain_table(
        all_df,
        maintenance,
        before_window_days=before_window_days,
        after_window_days=after_window_days,
    )
    gain_detail.to_csv(TABLE_DIR / "维护前后提升明细.csv", index=False, encoding="utf-8-sig")
    gain_summary.to_csv(TABLE_DIR / "维护前后提升量汇总表.csv", index=False, encoding="utf-8-sig")
    save_table_as_png(gain_summary, "维护前后提升量汇总表", PLOT_DIR / "05_maintenance_gain_summary.png")

    # 5. 季节性箱线图
    plot_season_boxplot(all_df)

    # 6. 自然下降速率表
    decline_table = calc_natural_decline_rate(
        all_df,
        maintenance,
        exclude_window_days=exclude_window_days,
    )
    decline_table.to_csv(TABLE_DIR / "自然下降速率表.csv", index=False, encoding="utf-8-sig")
    save_table_as_png(decline_table, "自然下降速率表", PLOT_DIR / "06_natural_decline_rate.png")

    print("[完成] A题数据分析输出已生成")
    print(f"- 逐设备透水率曲线：{DEVICE_CURVE_DIR}")
    print(f"- 逐设备维护点标注图：{MAINTAIN_PLOT_DIR}")
    print(f"- 汇总图片：{PLOT_DIR}")
    print(f"- CSV表格：{TABLE_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A题过滤设备数据清洗、异常检查、可视化与初步指标计算")

    parser.add_argument(
        "--missing-strategy",
        choices=["mean", "drop"],
        default="mean",
        help="透水率缺失值处理方式：mean=按同一设备均值填充；drop=删除缺失记录。默认 mean。",
    )
    parser.add_argument(
        "--treat-outliers-as-missing",
        action="store_true",
        help="将IQR异常值也视作缺失值，再按 --missing-strategy 处理。默认只记录异常值，不改动。",
    )
    parser.add_argument(
        "--before-window-days",
        type=int,
        default=2,
        help="计算维护前透水率时，向前搜索的天数窗口。默认2天。",
    )
    parser.add_argument(
        "--after-window-days",
        type=int,
        default=2,
        help="计算维护后透水率时，向后搜索的天数窗口。默认2天。",
    )
    parser.add_argument(
        "--exclude-window-days",
        type=int,
        default=3,
        help="计算自然下降速率时，排除维护日前后多少天。默认3天。",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        missing_strategy=args.missing_strategy,
        treat_outliers_as_missing=args.treat_outliers_as_missing,
        before_window_days=args.before_window_days,
        after_window_days=args.after_window_days,
        exclude_window_days=args.exclude_window_days,
    )
