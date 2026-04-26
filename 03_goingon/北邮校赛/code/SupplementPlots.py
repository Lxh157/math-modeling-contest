# python ./03_goingon/北邮校赛/code/SupplementPlots.py

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D


# =========================
# 路径配置
# =========================
CODE_DIR = Path(__file__).resolve().parent
BASE_DIR = CODE_DIR.parent
RAW_DATA_DIR = BASE_DIR / "data" / "原始数据"
PLOT_DIR = BASE_DIR / "plot"
TABLE_DIR = PLOT_DIR / "tables"


# =========================
# 基础工具
# =========================
def configure_chinese_font() -> None:
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
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def season_label(month: int) -> str:
    if month in (3, 4, 5):
        return "春"
    if month in (6, 7, 8):
        return "夏"
    if month in (9, 10, 11):
        return "秋"
    return "冬"


def normalize_device_code(value: object) -> str:
    text = str(value).strip().upper().replace("_", "")
    if text.startswith("A"):
        suffix = text[1:]
        if suffix.isdigit():
            return f"A{int(suffix)}"
    return text


def find_existing_file(candidates: Iterable[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"未找到候选文件：{[str(p) for p in candidates]}")


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise KeyError(f"在列 {list(df.columns)} 中未找到任何候选列：{candidates}")


def save_table_as_png(df: pd.DataFrame, title: str, output_path: Path, max_rows: int | None = None) -> None:
    show_df = df.copy()
    if max_rows is not None and len(show_df) > max_rows:
        show_df = show_df.head(max_rows)

    fig_height = max(3.0, min(18, 0.42 * len(show_df) + 1.8))
    fig_width = max(10, min(22, 1.45 * len(show_df.columns) + 2.5))

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
def load_clean_long_df() -> pd.DataFrame:
    path = find_existing_file([
        TABLE_DIR / "清洗后透水率长表.csv",
        TABLE_DIR / "clean_long_table.csv",
    ])
    df = pd.read_csv(path)

    device_col = find_column(df, ["device_code", "设备编号", "device", "编号"])
    time_col = find_column(df, ["time", "时间", "date", "检测时间"])
    per_col = find_column(df, ["per", "透水率", "permeability", "透水率值"])

    df = df.rename(columns={
        device_col: "device_code",
        time_col: "time",
        per_col: "per",
    })

    df["device_code"] = df["device_code"].map(normalize_device_code)
    df["time"] = pd.to_datetime(df["time"])
    df["per"] = pd.to_numeric(df["per"], errors="coerce")
    df = df.dropna(subset=["device_code", "time", "per"]).copy()

    if "month" not in df.columns:
        df["month"] = df["time"].dt.month
    if "season" not in df.columns:
        df["season"] = df["month"].map(season_label)

    df = df.sort_values(["device_code", "time"]).reset_index(drop=True)
    return df


def load_maintenance_gain_detail() -> pd.DataFrame:
    path = find_existing_file([
        TABLE_DIR / "维护前后提升明细.csv",
        TABLE_DIR / "maintenance_gain_detail.csv",
    ])
    df = pd.read_csv(path)

    rename_map = {}
    rename_map[find_column(df, ["device_code", "设备编号", "device", "编号"])] = "device_code"
    rename_map[find_column(df, ["维护类型", "maintenance_type", "type"])] = "maintenance_type"

    # 日期列：尽量兼容
    try:
        rename_map[find_column(df, ["维护日期", "maintenance_date", "date"])] = "maintenance_date"
    except KeyError:
        pass

    # 维护前透水率、维护后透水率、提升量
    rename_map[find_column(df, ["维护前透水率", "维护前", "per_before", "before_per", "pre_per"])] = "per_before"
    rename_map[find_column(df, ["维护后透水率", "维护后", "per_after", "after_per", "post_per"])] = "per_after"
    rename_map[find_column(df, ["提升量", "透水率提升量", "gain", "delta_per"])] = "gain"

    df = df.rename(columns=rename_map)

    df["device_code"] = df["device_code"].map(normalize_device_code)
    if "maintenance_date" in df.columns:
        df["maintenance_date"] = pd.to_datetime(df["maintenance_date"], errors="coerce")
    df["per_before"] = pd.to_numeric(df["per_before"], errors="coerce")
    df["per_after"] = pd.to_numeric(df["per_after"], errors="coerce")
    df["gain"] = pd.to_numeric(df["gain"], errors="coerce")
    df["maintenance_type"] = df["maintenance_type"].astype(str).str.strip()

    df = df.dropna(subset=["device_code", "maintenance_type", "gain"]).copy()
    return df


def load_maintenance_log() -> pd.DataFrame:
    # 优先尝试原始附件2.xlsx
    xlsx_candidates = [
        RAW_DATA_DIR / "附件2.xlsx",
        RAW_DATA_DIR / "附件2.xls",
        RAW_DATA_DIR / "附件2_维护记录.xlsx",
    ]
    csv_candidates = [
        RAW_DATA_DIR / "附件2.csv",
        RAW_DATA_DIR / "附件2__维护记录.csv",
        RAW_DATA_DIR / "维护记录.csv",
    ]

    maintenance_df = None

    for path in xlsx_candidates:
        if path.exists():
            maintenance_df = pd.read_excel(path)
            break

    if maintenance_df is None:
        for path in csv_candidates:
            if path.exists():
                maintenance_df = pd.read_csv(path)
                break

    if maintenance_df is None:
        raise FileNotFoundError("未找到附件2维护记录文件（xlsx/csv）。")

    device_col = find_column(maintenance_df, ["编号", "设备编号", "device_code", "device"])
    date_col = find_column(maintenance_df, ["日期", "维护日期", "date", "maintenance_date"])
    type_col = find_column(maintenance_df, ["维护类型", "maintenance_type", "type"])

    maintenance_df = maintenance_df.rename(columns={
        device_col: "device_code",
        date_col: "maintenance_date",
        type_col: "maintenance_type",
    })
    maintenance_df["device_code"] = maintenance_df["device_code"].map(normalize_device_code)
    maintenance_df["maintenance_date"] = pd.to_datetime(maintenance_df["maintenance_date"], errors="coerce")
    maintenance_df["maintenance_type"] = maintenance_df["maintenance_type"].astype(str).str.strip()

    maintenance_df = maintenance_df.dropna(subset=["device_code", "maintenance_date", "maintenance_type"]).copy()
    maintenance_df = maintenance_df.sort_values(["device_code", "maintenance_date"]).reset_index(drop=True)
    return maintenance_df


# =========================
# 图 1：月度均值/中位数折线图
# =========================
def plot_monthly_mean_trend(clean_df: pd.DataFrame) -> pd.DataFrame:
    monthly_stats = (
        clean_df.groupby("month")["per"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .sort_values("month")
    )
    monthly_stats["month_name"] = monthly_stats["month"].astype(int).astype(str)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(monthly_stats["month"], monthly_stats["mean"], marker="o", linewidth=2, label="月均值")
    ax.plot(monthly_stats["month"], monthly_stats["median"], marker="s", linewidth=2, label="月中位数")

    ax.set_title("各月份透水率均值/中位数变化趋势")
    ax.set_xlabel("月份")
    ax.set_ylabel("透水率")
    ax.set_xticks(range(1, 13))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "07_monthly_mean_trend.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    monthly_stats.to_csv(TABLE_DIR / "月度统计表.csv", index=False, encoding="utf-8-sig")
    return monthly_stats


# =========================
# 图 2：设备综合特征表（PNG + CSV）
# =========================
def build_device_feature_table(clean_df: pd.DataFrame, gain_df: pd.DataFrame, seg_summary_df: pd.DataFrame) -> pd.DataFrame:
    basic = (
        clean_df.groupby("device_code")["per"]
        .agg(["count", "mean", "median", "min", "max", "std"])
        .reset_index()
        .rename(columns={
            "count": "清洗后样本数",
            "mean": "平均透水率",
            "median": "中位透水率",
            "min": "最小透水率",
            "max": "最大透水率",
            "std": "标准差",
        })
    )
    basic["变异系数CV"] = basic["标准差"] / basic["平均透水率"]

    # 维护次数
    count_table = (
        gain_df.pivot_table(
            index="device_code",
            columns="maintenance_type",
            values="gain",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )
    count_table.columns = [str(c) for c in count_table.columns]
    if "中维护" not in count_table.columns:
        count_table["中维护"] = 0
    if "大维护" not in count_table.columns:
        count_table["大维护"] = 0
    count_table = count_table.rename(columns={
        "中维护": "中维护次数",
        "大维护": "大维护次数",
    })

    # 平均维护提升量、维护有效率
    gain_summary = (
        gain_df.groupby("device_code")
        .agg(
            平均维护提升量=("gain", "mean"),
            维护提升量中位数=("gain", "median"),
            维护有效率=("gain", lambda s: float((s > 0).mean()) if len(s) > 0 else np.nan),
        )
        .reset_index()
    )

    seg_part = seg_summary_df[[
        "device_code",
        "有效区段数",
        "负斜率区段数",
        "平均区段下降速率(每天)",
        "平均区段下降速率(每年)",
    ]].copy()

    result = basic.merge(count_table, on="device_code", how="left")
    result = result.merge(gain_summary, on="device_code", how="left")
    result = result.merge(seg_part, on="device_code", how="left")

    for col in ["中维护次数", "大维护次数", "有效区段数", "负斜率区段数"]:
        if col in result.columns:
            result[col] = result[col].fillna(0).astype(int)

    # 格式美化
    round_cols = [
        "平均透水率", "中位透水率", "最小透水率", "最大透水率", "标准差", "变异系数CV",
        "平均维护提升量", "维护提升量中位数", "维护有效率",
        "平均区段下降速率(每天)", "平均区段下降速率(每年)",
    ]
    for col in round_cols:
        if col in result.columns:
            result[col] = result[col].round(4)

    result = result.sort_values("device_code").reset_index(drop=True)
    result.to_csv(TABLE_DIR / "设备综合特征表.csv", index=False, encoding="utf-8-sig")
    save_table_as_png(result, "设备综合特征表", PLOT_DIR / "08_device_feature_table.png")
    return result


# =========================
# 图 3：按设备分组的维护提升量箱线图
# =========================
def plot_maintenance_gain_by_device(gain_df: pd.DataFrame) -> None:
    devices = sorted(gain_df["device_code"].dropna().unique(), key=lambda x: int(x[1:]))
    types = ["中维护", "大维护"]
    colors = {
        "中维护": "#1f77b4",
        "大维护": "#ff7f0e",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    base_positions = np.arange(len(devices)) * 3.0
    offsets = {"中维护": -0.45, "大维护": 0.45}
    width = 0.7

    for t in types:
        data_list = []
        positions = []
        for i, dev in enumerate(devices):
            vals = gain_df.loc[
                (gain_df["device_code"] == dev) & (gain_df["maintenance_type"] == t),
                "gain"
            ].dropna().values
            if len(vals) == 0:
                vals = np.array([np.nan])
            data_list.append(vals)
            positions.append(base_positions[i] + offsets[t])

        bp = ax.boxplot(
            data_list,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=True,
            medianprops=dict(color="black", linewidth=1.2),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[t])
            patch.set_alpha(0.55)
        for item in ["whiskers", "caps"]:
            for line in bp[item]:
                line.set_color(colors[t])

    ax.set_xticks(base_positions)
    ax.set_xticklabels(devices)
    ax.set_title("按设备分组的维护提升量箱线图")
    ax.set_xlabel("设备编号")
    ax.set_ylabel("维护提升量")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    legend_elements = [
        Line2D([0], [0], color=colors["中维护"], lw=8, alpha=0.55, label="中维护"),
        Line2D([0], [0], color=colors["大维护"], lw=8, alpha=0.55, label="大维护"),
    ]
    ax.legend(handles=legend_elements)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "09_maintenance_gain_by_device.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================
# 图 4：维护前透水率 vs 提升量散点图
# =========================
def plot_before_per_vs_gain(gain_df: pd.DataFrame) -> None:
    df = gain_df.dropna(subset=["per_before", "gain"]).copy()

    colors = {
        "中维护": "#1f77b4",
        "大维护": "#ff7f0e",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for t in ["中维护", "大维护"]:
        sub = df[df["maintenance_type"] == t].copy()
        if sub.empty:
            continue

        ax.scatter(
            sub["per_before"],
            sub["gain"],
            s=36,
            alpha=0.70,
            label=t,
            color=colors[t],
        )

        if len(sub) >= 2:
            x = sub["per_before"].values
            y = sub["gain"].values
            coef = np.polyfit(x, y, deg=1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = coef[0] * x_line + coef[1]
            ax.plot(x_line, y_line, linestyle="--", linewidth=2, color=colors[t])

    ax.set_title("维护前透水率与维护提升量关系图")
    ax.set_xlabel("维护前透水率")
    ax.set_ylabel("维护提升量")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "10_before_permeability_vs_gain.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================
# 图 5：分维护间隔段下降速率（表 + 图）
# =========================
def compute_segment_decline_rates(clean_df: pd.DataFrame, maintenance_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    details = []

    devices = sorted(clean_df["device_code"].unique(), key=lambda x: int(x[1:]))

    for dev in devices:
        dev_df = clean_df[clean_df["device_code"] == dev].sort_values("time").copy()
        if dev_df.empty:
            continue

        dev_maint = maintenance_df[maintenance_df["device_code"] == dev].sort_values("maintenance_date").copy()
        maint_dates = list(dev_maint["maintenance_date"].dropna().sort_values())

        start_time = dev_df["time"].min()
        end_time = dev_df["time"].max()

        # 用相邻维护事件将序列切段
        boundaries = [start_time] + maint_dates + [end_time]

        for idx in range(len(boundaries) - 1):
            left = boundaries[idx]
            right = boundaries[idx + 1]

            # 若 left / right 是维护日期，则各自避开前后3天扰动
            seg_start = left + pd.Timedelta(days=3) if idx > 0 else left
            seg_end = right - pd.Timedelta(days=3) if idx < len(boundaries) - 2 else right

            seg_df = dev_df[(dev_df["time"] >= seg_start) & (dev_df["time"] <= seg_end)].copy()
            seg_df = seg_df.sort_values("time")

            if seg_df.empty:
                continue

            unique_days = (seg_df["time"].dt.floor("D").nunique())
            if unique_days < 7 or len(seg_df) < 5:
                continue

            x = (seg_df["time"] - seg_df["time"].min()).dt.total_seconds() / 86400.0
            y = seg_df["per"].values

            if len(np.unique(x)) < 2:
                continue

            slope, intercept = np.polyfit(x, y, 1)

            # R^2
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

            details.append({
                "device_code": dev,
                "区段编号": idx + 1,
                "区段起点": seg_df["time"].min(),
                "区段终点": seg_df["time"].max(),
                "样本数": len(seg_df),
                "覆盖天数": round(float(x.max() - x.min()), 4),
                "下降速率(每天)": float(slope),
                "下降速率(每年)": float(slope * 365),
                "截距": float(intercept),
                "R2": float(r2) if pd.notna(r2) else np.nan,
                "是否负斜率": int(slope < 0),
            })

    detail_df = pd.DataFrame(details)
    if detail_df.empty:
        raise ValueError("未能计算出任何维护间隔段下降速率，请检查维护记录与清洗后长表。")

    summary_df = (
        detail_df.groupby("device_code")
        .agg(
            有效区段数=("区段编号", "count"),
            负斜率区段数=("是否负斜率", "sum"),
            平均区段斜率_全部=("下降速率(每天)", "mean"),
            平均区段斜率_负值=("下降速率(每天)", lambda s: s[s < 0].mean() if (s < 0).any() else np.nan),
            中位区段斜率_负值=("下降速率(每天)", lambda s: s[s < 0].median() if (s < 0).any() else np.nan),
            平均R2=("R2", "mean"),
        )
        .reset_index()
    )

    # 优先使用“负斜率平均值”作为自然下降速率；若该设备没有负斜率段，则退回全部平均
    summary_df["平均区段下降速率(每天)"] = summary_df["平均区段斜率_负值"].combine_first(summary_df["平均区段斜率_全部"])
    summary_df["平均区段下降速率(每年)"] = summary_df["平均区段下降速率(每天)"] * 365

    keep_cols = [
        "device_code",
        "有效区段数",
        "负斜率区段数",
        "平均区段下降速率(每天)",
        "平均区段下降速率(每年)",
        "中位区段斜率_负值",
        "平均R2",
    ]
    summary_df = summary_df[keep_cols].copy()

    round_cols = [
        "平均区段下降速率(每天)",
        "平均区段下降速率(每年)",
        "中位区段斜率_负值",
        "平均R2",
    ]
    for col in round_cols:
        summary_df[col] = summary_df[col].round(6)

    detail_df = detail_df.sort_values(["device_code", "区段编号"]).reset_index(drop=True)
    summary_df = summary_df.sort_values("device_code").reset_index(drop=True)

    detail_df.to_csv(TABLE_DIR / "维护间隔段下降速率明细.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(TABLE_DIR / "维护间隔段下降速率表.csv", index=False, encoding="utf-8-sig")
    return detail_df, summary_df


def plot_segment_decline_rate(summary_df: pd.DataFrame) -> None:
    df = summary_df.sort_values("device_code").copy()
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.bar(x, df["平均区段下降速率(每年)"], alpha=0.75)

    for rect, cnt in zip(bars, df["有效区段数"]):
        height = rect.get_height()
        va = "bottom" if height >= 0 else "top"
        offset = 0.6 if height >= 0 else -0.6
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            height + offset,
            f"n={cnt}",
            ha="center",
            va=va,
            fontsize=9,
        )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["device_code"])
    ax.set_title("基于维护间隔段的设备自然下降速率（年化）")
    ax.set_xlabel("设备编号")
    ax.set_ylabel("平均区段下降速率（每年）")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "11_segment_decline_rate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================
# 主函数
# =========================
def main() -> None:
    configure_chinese_font()
    ensure_output_dirs()

    print("读取清洗后透水率长表...")
    clean_df = load_clean_long_df()

    print("读取维护前后提升明细...")
    gain_df = load_maintenance_gain_detail()

    print("读取维护记录...")
    maintenance_df = load_maintenance_log()

    print("生成：月度均值/中位数折线图...")
    plot_monthly_mean_trend(clean_df)

    print("生成：维护间隔段下降速率表与图...")
    detail_df, seg_summary_df = compute_segment_decline_rates(clean_df, maintenance_df)
    plot_segment_decline_rate(seg_summary_df)

    print("生成：设备综合特征表...")
    build_device_feature_table(clean_df, gain_df, seg_summary_df)

    print("生成：按设备分组的维护提升量箱线图...")
    plot_maintenance_gain_by_device(gain_df)

    print("生成：维护前透水率 vs 提升量散点图...")
    plot_before_per_vs_gain(gain_df)

    print("全部完成。新增输出如下：")
    print(f"- {PLOT_DIR / '07_monthly_mean_trend.png'}")
    print(f"- {PLOT_DIR / '08_device_feature_table.png'}")
    print(f"- {PLOT_DIR / '09_maintenance_gain_by_device.png'}")
    print(f"- {PLOT_DIR / '10_before_permeability_vs_gain.png'}")
    print(f"- {PLOT_DIR / '11_segment_decline_rate.png'}")
    print(f"- {TABLE_DIR / '月度统计表.csv'}")
    print(f"- {TABLE_DIR / '设备综合特征表.csv'}")
    print(f"- {TABLE_DIR / '维护间隔段下降速率明细.csv'}")
    print(f"- {TABLE_DIR / '维护间隔段下降速率表.csv'}")


if __name__ == "__main__":
    main()