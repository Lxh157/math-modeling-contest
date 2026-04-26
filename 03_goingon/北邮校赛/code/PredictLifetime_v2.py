from __future__ import annotations

"""
A题：过滤设备监测 - 第2问寿命预测脚本（改进版）
建议放置位置：
03_goingon/北邮校赛/code/PredictLifetime_v2.py

运行方式（仓库根目录）：
python ./03_goingon/北邮校赛/code/PredictLifetime_v2.py
"""

import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


CODE_DIR = Path(__file__).resolve().parent
BASE_DIR = CODE_DIR.parent
RAW_DATA_DIR = BASE_DIR / "data" / "原始数据"
PLOT_DIR = BASE_DIR / "plot"
TABLE_DIR = PLOT_DIR / "tables"


def configure_chinese_font() -> None:
    candidates = [
        "Microsoft YaHei", "SimHei", "Microsoft JhengHei",
        "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS",
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


def normalize_device_code(value: object) -> str:
    text = str(value).strip().upper().replace("_", "")
    if text.startswith("A"):
        suffix = text[1:]
        if suffix.isdigit():
            return f"A{int(suffix)}"
    return text


def find_existing_file(candidates: list[Path]) -> Path:
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


def add_common_time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30)


def season_label(month: int) -> str:
    if month in (3, 4, 5):
        return "春"
    if month in (6, 7, 8):
        return "夏"
    if month in (9, 10, 11):
        return "秋"
    return "冬"


def assign_gain_bucket(p_before: float) -> str:
    if p_before < 60:
        return "low(<60)"
    if p_before < 80:
        return "mid([60,80))"
    return "high(>=80)"


def load_clean_long_df() -> pd.DataFrame:
    path = find_existing_file([TABLE_DIR / "清洗后透水率长表.csv"])
    df = pd.read_csv(path)

    device_col = find_column(df, ["device_code", "设备编号", "device", "编号"])
    time_col = find_column(df, ["time", "时间", "日期", "date", "检测时间"])
    per_col = find_column(df, ["per", "透水率", "permeability", "透水率值"])

    rename_map = {device_col: "device_code", time_col: "time", per_col: "per"}
    if "月份" in df.columns:
        rename_map["月份"] = "month"
    if "季节" in df.columns:
        rename_map["季节"] = "season"

    df = df.rename(columns=rename_map)
    df["device_code"] = df["device_code"].map(normalize_device_code)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["per"] = pd.to_numeric(df["per"], errors="coerce")
    df = df.dropna(subset=["device_code", "time", "per"]).copy()

    if "month" not in df.columns:
        df["month"] = df["time"].dt.month
    if "season" not in df.columns:
        df["season"] = df["month"].map(season_label)

    return df.sort_values(["device_code", "time"]).reset_index(drop=True)


def load_segment_decline_summary() -> pd.DataFrame:
    path = find_existing_file([TABLE_DIR / "维护间隔段下降速率表.csv"])
    df = pd.read_csv(path)
    df = df.rename(columns={
        find_column(df, ["device_code", "设备编号", "device"]): "device_code",
        find_column(df, ["平均区段下降速率(每天)", "平均区段下降速率_每天", "daily_decline_rate"]): "daily_decline_rate",
        find_column(df, ["平均区段下降速率(每年)", "annual_decline_rate", "平均区段下降速率_每年"]): "annual_decline_rate",
    })
    df["device_code"] = df["device_code"].map(normalize_device_code)
    df["daily_decline_rate"] = pd.to_numeric(df["daily_decline_rate"], errors="coerce")
    df["annual_decline_rate"] = pd.to_numeric(df["annual_decline_rate"], errors="coerce")
    return df.sort_values("device_code").reset_index(drop=True)


def load_maintenance_gain_detail() -> pd.DataFrame:
    path = find_existing_file([TABLE_DIR / "维护前后提升明细.csv"])
    df = pd.read_csv(path)

    rename_map = {
        find_column(df, ["device_code", "设备编号", "device", "编号"]): "device_code",
        find_column(df, ["维护类型", "maintenance_type", "type"]): "maintenance_type",
        find_column(df, ["维护前透水率", "维护前", "before_per", "per_before", "pre_per"]): "per_before",
        find_column(df, ["维护后透水率", "维护后", "after_per", "per_after", "post_per"]): "per_after",
        find_column(df, ["提升量", "透水率提升量", "gain", "delta_per", "delta"]): "gain",
    }
    if any(c in df.columns for c in ["维护日期", "maintenance_date", "date"]):
        rename_map[find_column(df, ["维护日期", "maintenance_date", "date"])] = "maintenance_date"

    df = df.rename(columns=rename_map)
    df["device_code"] = df["device_code"].map(normalize_device_code)
    if "maintenance_date" in df.columns:
        df["maintenance_date"] = pd.to_datetime(df["maintenance_date"], errors="coerce")
    df["per_before"] = pd.to_numeric(df["per_before"], errors="coerce")
    df["per_after"] = pd.to_numeric(df["per_after"], errors="coerce")
    df["gain"] = pd.to_numeric(df["gain"], errors="coerce")
    df["maintenance_type"] = df["maintenance_type"].astype(str).str.strip()
    df = df.dropna(subset=["device_code", "maintenance_type", "gain", "per_before"]).copy()
    return df.sort_values(["device_code", "maintenance_type"]).reset_index(drop=True)


def load_maintenance_log() -> pd.DataFrame:
    xlsx_candidates = [RAW_DATA_DIR / "附件2.xlsx", RAW_DATA_DIR / "附件2.xls"]
    csv_candidates = [RAW_DATA_DIR / "附件2.csv", RAW_DATA_DIR / "维护记录.csv"]

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
    return maintenance_df.sort_values(["device_code", "maintenance_date"]).reset_index(drop=True)


def extract_latest_baseline(clean_df: pd.DataFrame) -> pd.DataFrame:
    latest = (
        clean_df.sort_values(["device_code", "time"])
        .groupby("device_code")
        .tail(1)[["device_code", "time", "per"]]
        .rename(columns={"time": "predict_start_time", "per": "predict_start_per"})
        .sort_values("device_code")
        .reset_index(drop=True)
    )
    return latest


def build_season_adjustment_detrended(clean_df: pd.DataFrame, decline_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = clean_df.merge(decline_df[["device_code", "daily_decline_rate"]], on="device_code", how="left").copy()
    if df["daily_decline_rate"].isna().any():
        missing = df.loc[df["daily_decline_rate"].isna(), "device_code"].unique().tolist()
        raise ValueError(f"以下设备缺少下降速率，无法构造去趋势季节项：{missing}")

    df["days_from_start"] = (
        df["time"] - df.groupby("device_code")["time"].transform("min")
    ).dt.total_seconds() / 86400.0

    device_mean = df.groupby("device_code")["per"].transform("mean")
    df["trend_component"] = df["daily_decline_rate"] * df["days_from_start"]
    df["residual_detrended"] = df["per"] - (device_mean + df["trend_component"])

    season_table = (
        df.groupby("season")["residual_detrended"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "season_adjustment"})
    )
    season_map = {row["season"]: float(row["season_adjustment"]) for _, row in season_table.iterrows()}
    return season_table, season_map


def compute_segmented_maintenance_gains(gain_df: pd.DataFrame) -> pd.DataFrame:
    gdf = gain_df.copy()
    gdf["gain_bucket"] = gdf["per_before"].apply(assign_gain_bucket)
    result = (
        gdf.groupby(["maintenance_type", "gain_bucket"])["gain"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "avg_gain",
            "median": "median_gain",
            "std": "std_gain",
            "count": "sample_count",
        })
        .sort_values(["maintenance_type", "gain_bucket"])
        .reset_index(drop=True)
    )
    return result


def build_gain_lookup(seg_gain_df: pd.DataFrame) -> dict:
    lookup = {}
    for _, row in seg_gain_df.iterrows():
        lookup[(row["maintenance_type"], row["gain_bucket"])] = float(row["avg_gain"])
    return lookup


def compute_annual_maintenance_frequency(maintenance_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for device, dev_df in maintenance_df.groupby("device_code"):
        dev_df = dev_df.copy()
        span_days = (dev_df["maintenance_date"].max() - dev_df["maintenance_date"].min()).days
        span_years = max(span_days / 365.0, 1.0)

        rec = {"device_code": device}
        for mtype, out_name in [("中维护", "mid_freq_per_year"), ("大维护", "big_freq_per_year")]:
            cnt = int((dev_df["maintenance_type"] == mtype).sum())
            rec[out_name] = cnt / span_years
            rec[f"{mtype}_count_total"] = cnt
        records.append(rec)

    return pd.DataFrame(records).sort_values("device_code").reset_index(drop=True)


def build_parameter_table(clean_df: pd.DataFrame, decline_df: pd.DataFrame, maintenance_df: pd.DataFrame) -> pd.DataFrame:
    latest_df = extract_latest_baseline(clean_df)
    freq_df = compute_annual_maintenance_frequency(maintenance_df)

    param_df = latest_df.merge(
        decline_df[["device_code", "daily_decline_rate", "annual_decline_rate"]],
        on="device_code", how="left"
    )
    param_df = param_df.merge(freq_df, on="device_code", how="left")

    param_df["mid_freq_per_year"] = param_df["mid_freq_per_year"].fillna(param_df["mid_freq_per_year"].mean())
    if param_df["big_freq_per_year"].dropna().empty:
        param_df["big_freq_per_year"] = 1.0
    else:
        param_df["big_freq_per_year"] = param_df["big_freq_per_year"].fillna(param_df["big_freq_per_year"].mean())

    param_df["mid_freq_per_year"] = param_df["mid_freq_per_year"].clip(lower=0.1)
    param_df["big_freq_per_year"] = param_df["big_freq_per_year"].clip(lower=0.1)

    if param_df["daily_decline_rate"].isna().any():
        missing = param_df.loc[param_df["daily_decline_rate"].isna(), "device_code"].tolist()
        raise ValueError(f"以下设备缺少下降速率：{missing}")

    return param_df.sort_values("device_code").reset_index(drop=True)


def generate_yearly_schedule(start_date: pd.Timestamp, years: int, mid_freq_per_year: float, big_freq_per_year: float) -> tuple[set[pd.Timestamp], set[pd.Timestamp]]:
    mid_dates = set()
    big_dates = set()

    for y in range(years):
        year_start = start_date + pd.Timedelta(days=365 * y)

        n_mid = max(1, int(round(mid_freq_per_year)))
        mid_offsets = np.linspace(365 / (n_mid + 1), 365 - 365 / (n_mid + 1), n_mid)
        for off in mid_offsets:
            mid_dates.add((year_start + pd.Timedelta(days=float(off))).normalize())

        n_big = max(1, int(round(big_freq_per_year)))
        big_offsets = np.linspace(365 / (n_big + 1), 365 - 365 / (n_big + 1), n_big)
        for off in big_offsets:
            big_dates.add((year_start + pd.Timedelta(days=float(off))).normalize())

    return mid_dates, big_dates


def season_adjustment_for(season: str, season_map: dict) -> float:
    return float(season_map.get(season, 0.0))


def maintenance_gain_for(maintenance_type: str, p_before: float, gain_lookup: dict) -> float:
    bucket = assign_gain_bucket(p_before)
    if (maintenance_type, bucket) in gain_lookup:
        return float(gain_lookup[(maintenance_type, bucket)])

    type_values = [v for (k_type, _), v in gain_lookup.items() if k_type == maintenance_type]
    if type_values:
        return float(np.mean(type_values))
    return 0.0


def simulate_device_trajectory_v2(row: pd.Series, season_map: dict, gain_lookup: dict, horizon_years: int = 10, failure_threshold: float = 37.0) -> tuple[pd.DataFrame, dict]:
    device = row["device_code"]
    start_date = pd.to_datetime(row["predict_start_time"]).normalize()
    current_per = float(row["predict_start_per"])
    daily_decline_rate = float(row["daily_decline_rate"])
    mid_freq_per_year = float(row["mid_freq_per_year"])
    big_freq_per_year = float(row["big_freq_per_year"])

    total_days = int(horizon_years * 365)
    mid_schedule, big_schedule = generate_yearly_schedule(start_date, horizon_years, mid_freq_per_year, big_freq_per_year)

    records = []
    failed_date = None
    failed_reason = ""

    for d in range(total_days + 1):
        current_date = start_date + pd.Timedelta(days=d)
        current_season = season_label(current_date.month)

        if d == 0:
            maintenance_type = "起点"
            gain_today = 0.0
        else:
            prev_date = current_date - pd.Timedelta(days=1)
            prev_season = season_label(prev_date.month)

            current_per += daily_decline_rate
            current_per += season_adjustment_for(current_season, season_map) - season_adjustment_for(prev_season, season_map)

            if current_date in big_schedule:
                p_before = current_per
                gain_today = maintenance_gain_for("大维护", p_before, gain_lookup)
                current_per += gain_today
                maintenance_type = "大维护"
            elif current_date in mid_schedule:
                p_before = current_per
                gain_today = maintenance_gain_for("中维护", p_before, gain_lookup)
                current_per += gain_today
                maintenance_type = "中维护"
            else:
                maintenance_type = "无"
                gain_today = 0.0

        records.append({
            "device_code": device,
            "date": current_date,
            "predicted_permeability": current_per,
            "maintenance_type": maintenance_type,
            "gain_today": gain_today,
            "season": current_season,
            "year_index": (current_date - start_date).days // 365 + 1,
        })

    traj_df = pd.DataFrame(records)

    year_summary = (
        traj_df[traj_df["year_index"] >= 1]
        .groupby("year_index")
        .agg(
            annual_mean=("predicted_permeability", "mean"),
            annual_min=("predicted_permeability", "min"),
            annual_max=("predicted_permeability", "max"),
        )
        .reset_index()
    )

    last_post_list = []
    for y in year_summary["year_index"]:
        sub = traj_df[traj_df["year_index"] == y]
        maint_sub = sub[sub["maintenance_type"].isin(["中维护", "大维护"])]
        if not maint_sub.empty:
            last_post = float(maint_sub.iloc[-1]["predicted_permeability"])
        else:
            last_post = np.nan
        last_post_list.append(last_post)
    year_summary["last_post_maintenance_per"] = last_post_list

    for _, yrow in year_summary.iterrows():
        if yrow["annual_mean"] < failure_threshold:
            post_val = yrow["last_post_maintenance_per"]
            if pd.isna(post_val) or post_val < failure_threshold:
                y = int(yrow["year_index"])
                failed_sub = traj_df[traj_df["year_index"] == y]
                failed_date = pd.to_datetime(failed_sub.iloc[-1]["date"])
                failed_reason = "年平均透水率低于37且当年最后一次计划维护后仍无法恢复"
                break

    if failed_date is None:
        failed_date = pd.to_datetime(traj_df.iloc[-1]["date"])
        failed_reason = "预测窗口内未达到失效阈值"

    summary = {
        "device_code": device,
        "predict_start_time": start_date,
        "predict_start_per": round(float(row["predict_start_per"]), 4),
        "daily_decline_rate": round(daily_decline_rate, 6),
        "annual_decline_rate": round(daily_decline_rate * 365, 6),
        "mid_freq_per_year": round(mid_freq_per_year, 4),
        "big_freq_per_year": round(big_freq_per_year, 4),
        "predicted_failure_date": failed_date,
        "remaining_life_days": int((failed_date - start_date).days),
        "failure_reason": failed_reason,
    }
    return traj_df, summary


def simulate_all_devices_v2(param_df: pd.DataFrame, season_map: dict, gain_lookup: dict, horizon_years: int = 10, failure_threshold: float = 37.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_traj = []
    all_summary = []

    for _, row in param_df.iterrows():
        traj_df, summary = simulate_device_trajectory_v2(
            row=row,
            season_map=season_map,
            gain_lookup=gain_lookup,
            horizon_years=horizon_years,
            failure_threshold=failure_threshold,
        )
        all_traj.append(traj_df)
        all_summary.append(summary)

    traj_all_df = pd.concat(all_traj, ignore_index=True)
    summary_df = pd.DataFrame(all_summary).sort_values("device_code").reset_index(drop=True)
    return traj_all_df, summary_df


def plot_all_prediction_curves_v2(traj_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 7.5))
    devices = sorted(traj_df["device_code"].unique(), key=lambda x: int(x[1:]))

    for dev in devices:
        sub = traj_df[traj_df["device_code"] == dev]
        ax.plot(sub["date"], sub["predicted_permeability"], linewidth=1.4, label=dev)

    ax.axhline(37, color="red", linestyle="--", linewidth=1.5, label="失效阈值 37")
    ax.set_title("10台过滤器未来透水率预测曲线（改进版）")
    ax.set_xlabel("日期")
    ax.set_ylabel("预测透水率")
    ax.grid(True, linestyle="--", alpha=0.30)
    add_common_time_axis(ax)
    ax.legend(ncol=3, fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "14_lifetime_prediction_curves_v2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_example_devices_v2(traj_df: pd.DataFrame, example_devices: list[str] | None = None) -> None:
    if example_devices is None:
        example_devices = ["A4", "A6", "A10"]

    fig, axes = plt.subplots(len(example_devices), 1, figsize=(13, 10), sharex=True)
    if len(example_devices) == 1:
        axes = [axes]

    for ax, dev in zip(axes, example_devices):
        sub = traj_df[traj_df["device_code"] == dev]
        ax.plot(sub["date"], sub["predicted_permeability"], linewidth=1.6)
        ax.axhline(37, color="red", linestyle="--", linewidth=1.2)

        maint_sub = sub[sub["maintenance_type"].isin(["中维护", "大维护"])]
        if not maint_sub.empty:
            colors = np.where(maint_sub["maintenance_type"] == "大维护", "#ff7f0e", "#1f77b4")
            ax.scatter(maint_sub["date"], maint_sub["predicted_permeability"], s=14, c=colors, alpha=0.75)

        ax.set_title(f"{dev} 未来透水率预测轨迹（改进版）")
        ax.set_ylabel("预测透水率")
        ax.grid(True, linestyle="--", alpha=0.30)

    axes[-1].set_xlabel("日期")
    add_common_time_axis(axes[-1])
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "15_lifetime_examples_v2_A4_A6_A10.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_chinese_font()
    ensure_output_dirs()

    print("读取第一问输出表格...")
    clean_df = load_clean_long_df()
    decline_df = load_segment_decline_summary()
    gain_df = load_maintenance_gain_detail()
    maintenance_df = load_maintenance_log()

    print("估计去趋势后的季节修正项...")
    season_table, season_map = build_season_adjustment_detrended(clean_df, decline_df)
    season_table.to_csv(TABLE_DIR / "季节修正参数表_v2.csv", index=False, encoding="utf-8-sig")

    print("估计分段维护提升参数...")
    seg_gain_df = compute_segmented_maintenance_gains(gain_df)
    seg_gain_df.to_csv(TABLE_DIR / "分段维护提升参数表_v2.csv", index=False, encoding="utf-8-sig")
    gain_lookup = build_gain_lookup(seg_gain_df)

    print("提取寿命预测参数...")
    param_df = build_parameter_table(clean_df, decline_df, maintenance_df)
    param_out = param_df.copy()
    param_out["predict_start_time"] = pd.to_datetime(param_out["predict_start_time"]).dt.strftime("%Y-%m-%d")
    param_out.to_csv(TABLE_DIR / "维护规律参数表_v2.csv", index=False, encoding="utf-8-sig")

    print("开始逐设备模拟未来透水率与寿命...")
    traj_df, summary_df = simulate_all_devices_v2(
        param_df=param_df,
        season_map=season_map,
        gain_lookup=gain_lookup,
        horizon_years=10,
        failure_threshold=37.0,
    )

    traj_out = traj_df.copy()
    traj_out["date"] = pd.to_datetime(traj_out["date"]).dt.strftime("%Y-%m-%d")
    traj_out.to_csv(TABLE_DIR / "预测轨迹长表_v2.csv", index=False, encoding="utf-8-sig")

    summary_out = summary_df.copy()
    summary_out["predict_start_time"] = pd.to_datetime(summary_out["predict_start_time"]).dt.strftime("%Y-%m-%d")
    summary_out["predicted_failure_date"] = pd.to_datetime(summary_out["predicted_failure_date"]).dt.strftime("%Y-%m-%d")
    summary_out.to_csv(TABLE_DIR / "寿命预测结果表_v2.csv", index=False, encoding="utf-8-sig")

    print("生成图像...")
    plot_all_prediction_curves_v2(traj_df)
    plot_example_devices_v2(traj_df, example_devices=["A4", "A6", "A10"])

    print("完成。输出如下：")
    print(f"- {TABLE_DIR / '季节修正参数表_v2.csv'}")
    print(f"- {TABLE_DIR / '分段维护提升参数表_v2.csv'}")
    print(f"- {TABLE_DIR / '维护规律参数表_v2.csv'}")
    print(f"- {TABLE_DIR / '寿命预测结果表_v2.csv'}")
    print(f"- {TABLE_DIR / '预测轨迹长表_v2.csv'}")
    print(f"- {PLOT_DIR / '14_lifetime_prediction_curves_v2.png'}")
    print(f"- {PLOT_DIR / '15_lifetime_examples_v2_A4_A6_A10.png'}")


if __name__ == "__main__":
    main()
