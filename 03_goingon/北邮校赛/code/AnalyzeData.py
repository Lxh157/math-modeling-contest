from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager


CODE_DIR = Path(__file__).resolve().parent
BASE_DIR = CODE_DIR.parent
RAW_DATA_DIR = BASE_DIR / "data" / "原始数据"
PLOT_DIR = BASE_DIR / "plot"
OUTPUT_TABLE_DIR = PLOT_DIR / "tables"


def configure_chinese_font() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Microsoft JhengHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available:
            plt.rcParams["font.family"] = font_name
            return
    plt.rcParams["font.family"] = "DejaVu Sans"


configure_chinese_font()
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore", message=r"Glyph .* missing from font\(s\)")


@dataclass
class DeviceData:
    device_code: str
    device_name: str
    dataframe: pd.DataFrame


def normalize_device_name(name: str) -> str:
    return name.replace("_", "").upper()


def season_label(month: int) -> str:
    if month in (3, 4, 5):
        return "春"
    if month in (6, 7, 8):
        return "夏"
    if month in (9, 10, 11):
        return "秋"
    return "冬"


def load_device_data() -> list[DeviceData]:
    device_files = sorted(RAW_DATA_DIR.glob("附件1__A_*.csv"))
    if not device_files:
        raise FileNotFoundError(f"未找到设备透水率 CSV 文件：{RAW_DATA_DIR}")

    devices: list[DeviceData] = []
    for file_path in device_files:
        sheet_name = file_path.stem.split("__")[-1]
        device_code = normalize_device_name(sheet_name)
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        if "time" not in df.columns or "per" not in df.columns:
            raise ValueError(f"{file_path.name} 缺少 `time` 或 `per` 列")

        df = df.rename(columns={"time": "时间", "per": "透水率"})
        df["时间"] = pd.to_datetime(df["时间"], errors="coerce")
        df["透水率"] = pd.to_numeric(df["透水率"], errors="coerce")
        df = df.sort_values("时间").reset_index(drop=True)
        devices.append(DeviceData(device_code=device_code, device_name=sheet_name, dataframe=df))
    return devices


def load_maintenance_data() -> pd.DataFrame:
    file_path = RAW_DATA_DIR / "附件2.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"未找到维护记录文件：{file_path}")

    maintenance = pd.read_csv(file_path, encoding="utf-8-sig")
    expected_cols = {"编号", "日期", "维护类型"}
    if not expected_cols.issubset(set(maintenance.columns)):
        raise ValueError(f"附件2.csv 列名异常，当前列：{maintenance.columns.tolist()}")

    maintenance["编号"] = maintenance["编号"].astype(str).str.upper()
    maintenance["日期"] = pd.to_datetime(maintenance["日期"], errors="coerce")
    maintenance = maintenance.dropna(subset=["日期"]).sort_values(["编号", "日期"]).reset_index(drop=True)
    return maintenance


def clean_device_data(devices: list[DeviceData], missing_strategy: str) -> tuple[list[DeviceData], pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    cleaned_devices: list[DeviceData] = []

    for device in devices:
        df = device.dataframe.copy()
        original_rows = len(df)
        missing_before = int(df["透水率"].isna().sum())

        q1 = df["透水率"].quantile(0.25)
        q3 = df["透水率"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (df["透水率"] < lower) | (df["透水率"] > upper)
        outlier_count = int(outlier_mask.sum())

        # 异常值按缺失值处理，再统一填充或删除。
        df.loc[outlier_mask, "透水率"] = np.nan

        if missing_strategy == "mean":
            mean_value = df["透水率"].mean()
            df["透水率"] = df["透水率"].fillna(mean_value)
            df = df.dropna(subset=["时间"]).reset_index(drop=True)
            action = "均值填充"
        else:
            df = df.dropna(subset=["时间", "透水率"]).reset_index(drop=True)
            action = "删除缺失"

        missing_after = int(df["透水率"].isna().sum() + df["时间"].isna().sum())
        summary_rows.append(
            {
                "设备编号": device.device_code,
                "原始行数": original_rows,
                "异常值(IQR)数量": outlier_count,
                "缺失值数量(清洗前)": missing_before + outlier_count,
                "缺失值处理方式": action,
                "缺失值数量(清洗后)": missing_after,
                "清洗后行数": len(df),
            }
        )
        cleaned_devices.append(DeviceData(device.device_code, device.device_name, df))

    summary = pd.DataFrame(summary_rows).sort_values("设备编号")
    return cleaned_devices, summary


def build_combined_df(devices: list[DeviceData]) -> pd.DataFrame:
    merged = []
    for device in devices:
        df = device.dataframe.copy()
        df["设备编号"] = device.device_code
        merged.append(df)
    return pd.concat(merged, ignore_index=True)


def plot_device_curves(all_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(18, 16), sharex=True)
    axes = axes.flatten()

    for idx, device_code in enumerate(sorted(all_df["设备编号"].unique())):
        ax = axes[idx]
        device_df = all_df[all_df["设备编号"] == device_code]
        ax.plot(device_df["时间"], device_df["透水率"], linewidth=0.8, color="#1f77b4")
        ax.set_title(device_code)
        ax.set_ylabel("透水率")

    for ax in axes[10:]:
        ax.axis("off")

    fig.suptitle("10台设备透水率曲线", fontsize=16)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "10台设备透水率曲线.png", dpi=300)
    plt.close(fig)


def plot_maintenance_annotations(all_df: pd.DataFrame, maintenance: pd.DataFrame) -> None:
    color_map = {"中维护": "#ff7f0e", "大维护": "#d62728"}
    fig, axes = plt.subplots(5, 2, figsize=(18, 16), sharex=True)
    axes = axes.flatten()

    for idx, device_code in enumerate(sorted(all_df["设备编号"].unique())):
        ax = axes[idx]
        device_df = all_df[all_df["设备编号"] == device_code]
        maintain_df = maintenance[maintenance["编号"] == device_code]

        ax.plot(device_df["时间"], device_df["透水率"], linewidth=0.8, color="#1f77b4", label="透水率")
        for maintenance_type, group in maintain_df.groupby("维护类型"):
            y_values = np.interp(
                group["日期"].astype(np.int64),
                device_df["时间"].astype(np.int64),
                device_df["透水率"],
            )
            ax.scatter(
                group["日期"],
                y_values,
                s=16,
                color=color_map.get(maintenance_type, "#2ca02c"),
                alpha=0.9,
                label=maintenance_type,
            )
        ax.set_title(device_code)
        ax.set_ylabel("透水率")
        ax.legend(loc="best", fontsize=8)

    for ax in axes[10:]:
        ax.axis("off")

    fig.suptitle("维护点标注图", fontsize=16)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "维护点标注图.png", dpi=300)
    plt.close(fig)


def maintenance_gain_table(all_df: pd.DataFrame, maintenance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in maintenance.iterrows():
        device_code = row["编号"]
        date = row["日期"]
        m_type = row["维护类型"]

        device_df = all_df[all_df["设备编号"] == device_code]
        before = device_df[device_df["时间"] < date].tail(1)
        after = device_df[device_df["时间"] >= date].head(1)
        if before.empty or after.empty:
            continue

        before_value = float(before["透水率"].iloc[0])
        after_value = float(after["透水率"].iloc[0])
        rows.append(
            {
                "设备编号": device_code,
                "维护日期": date.date(),
                "维护类型": m_type,
                "维护前透水率": round(before_value, 4),
                "维护后透水率": round(after_value, 4),
                "提升量": round(after_value - before_value, 4),
            }
        )

    detail_df = pd.DataFrame(rows)
    detail_df.to_csv(OUTPUT_TABLE_DIR / "维护前后提升明细.csv", index=False, encoding="utf-8-sig")

    summary = (
        detail_df.groupby("维护类型")["提升量"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "样本数",
                "mean": "平均提升量",
                "median": "中位提升量",
                "min": "最小提升量",
                "max": "最大提升量",
            }
        )
    )
    summary = summary.round(4)
    summary.to_csv(OUTPUT_TABLE_DIR / "维护前后提升量表.csv", index=False, encoding="utf-8-sig")
    return summary


def plot_table(dataframe: pd.DataFrame, title: str, output_name: str) -> None:
    fig, ax = plt.subplots(figsize=(12, max(3, len(dataframe) * 0.6 + 1)))
    ax.axis("off")
    table = ax.table(
        cellText=dataframe.values,
        colLabels=dataframe.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    ax.set_title(title, fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / output_name, dpi=300)
    plt.close(fig)


def plot_season_boxplot(all_df: pd.DataFrame) -> None:
    season_order = ["春", "夏", "秋", "冬"]
    df = all_df.copy()
    df["季节"] = df["时间"].dt.month.map(season_label)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="季节", y="透水率", order=season_order, ax=ax)
    ax.set_title("季节性箱线图")
    ax.set_xlabel("季节")
    ax.set_ylabel("透水率")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "季节性箱线图.png", dpi=300)
    plt.close(fig)


def natural_decline_rate_table(all_df: pd.DataFrame, maintenance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    window_days = 3

    for device_code, device_df in all_df.groupby("设备编号"):
        df = device_df.sort_values("时间").copy()
        maintain_dates = maintenance.loc[maintenance["编号"] == device_code, "日期"].dropna()
        mask = pd.Series(True, index=df.index)

        for date in maintain_dates:
            lower = date - pd.Timedelta(days=window_days)
            upper = date + pd.Timedelta(days=window_days)
            mask &= ~df["时间"].between(lower, upper)

        natural_df = df[mask]
        if len(natural_df) < 2:
            continue

        x_days = (natural_df["时间"] - natural_df["时间"].min()).dt.total_seconds() / 86400.0
        slope, intercept = np.polyfit(x_days, natural_df["透水率"], 1)
        rows.append(
            {
                "设备编号": device_code,
                "自然下降速率(每天)": round(float(slope), 6),
                "自然下降速率(每30天)": round(float(slope) * 30, 6),
                "样本点数": int(len(natural_df)),
                "拟合截距": round(float(intercept), 4),
            }
        )

    table_df = pd.DataFrame(rows).sort_values("设备编号")
    table_df.to_csv(OUTPUT_TABLE_DIR / "自然下降速率表.csv", index=False, encoding="utf-8-sig")
    return table_df


def run_pipeline(missing_strategy: str) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    device_data = load_device_data()
    maintenance = load_maintenance_data()
    cleaned_devices, outlier_summary = clean_device_data(device_data, missing_strategy=missing_strategy)
    all_df = build_combined_df(cleaned_devices)

    outlier_summary.to_csv(OUTPUT_TABLE_DIR / "异常值与缺失值处理汇总.csv", index=False, encoding="utf-8-sig")
    plot_table(outlier_summary, "异常值检查与缺失值处理汇总", "异常值检查汇总表.png")

    plot_device_curves(all_df)
    plot_maintenance_annotations(all_df, maintenance)

    gain_summary = maintenance_gain_table(all_df, maintenance)
    plot_table(gain_summary, "维护前后提升量表", "维护前后提升量表.png")

    plot_season_boxplot(all_df)

    decline_table = natural_decline_rate_table(all_df, maintenance)
    plot_table(decline_table, "自然下降速率表", "自然下降速率表.png")

    print("分析完成，输出目录：")
    print(f"- 图片：{PLOT_DIR}")
    print(f"- 表格CSV：{OUTPUT_TABLE_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="设备透水率数据清洗与可视化分析")
    parser.add_argument(
        "--missing-strategy",
        choices=["mean", "drop"],
        default="mean",
        help="缺失值处理方式：mean=均值填充，drop=删除缺失值",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(missing_strategy=args.missing_strategy)
