from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

CODE_DIR = Path(__file__).resolve().parent
# 运行时可自行把 BASE_DIR 改成你的项目目录；也支持直接放到 code 下运行
PROJECT_BASE = Path.cwd() / '03_goingon' / '北邮校赛'
PLOT_DIR = PROJECT_BASE / 'plot'
TABLE_DIR = PLOT_DIR / 'tables'
OUT_DIR = PLOT_DIR / 'paper_tables'


def configure_chinese_font() -> None:
    candidates = [
        'Microsoft YaHei',
        'SimHei',
        'Microsoft JhengHei',
        'Noto Sans CJK SC',
        'Source Han Sans SC',
        'Arial Unicode MS',
    ]
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available_fonts:
            plt.rcParams['font.family'] = font_name
            break
    else:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    warnings.filterwarnings('ignore', message=r'Glyph .* missing from font\(s\)')


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def format_df(df: pd.DataFrame, round_cols: list[str] | None = None, digits: int = 4) -> pd.DataFrame:
    out = df.copy()
    if round_cols is None:
        round_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    for c in round_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f'{x:.{digits}f}' if pd.notna(x) and isinstance(x, (int, float)) else x)
    return out


def save_table_png(df: pd.DataFrame, title: str, out_path: Path, fontsize: int = 9, max_rows: int | None = None) -> None:
    show_df = df.copy()
    if max_rows is not None and len(show_df) > max_rows:
        show_df = show_df.head(max_rows)
    fig_h = max(2.8, min(18, 0.45 * len(show_df) + 1.6))
    fig_w = max(10, min(24, 1.5 * len(show_df.columns) + 3))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')
    ax.set_title(title, fontsize=14, pad=10)
    table = ax.table(
        cellText=show_df.astype(str).values,
        colLabels=show_df.columns,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, 1.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def build_missing_summary() -> None:
    path = TABLE_DIR / '异常值与缺失值处理汇总.csv'
    if not path.exists():
        return
    df = pd.read_csv(path)
    keep = [c for c in ['设备编号', '原始样本数', '透水率缺失数', '清洗后样本数', 'IDR异常值数'] if c in df.columns]
    if not keep:
        keep = df.columns.tolist()
    out = format_df(df[keep], digits=0)
    save_table_png(out, '数据清洗汇总表', OUT_DIR / '表1_数据清洗汇总表.png', fontsize=10)


def build_decline_summary() -> None:
    path = TABLE_DIR / '维护间隔段下降速率表.csv'
    if not path.exists():
        return
    df = pd.read_csv(path)
    keep = [c for c in ['device_code', '有效区段数', '负斜率区段数', '平均区段下降速率(每天)', '平均区段下降速率(每年)', '平均R2'] if c in df.columns]
    out = format_df(df[keep], digits=4)
    save_table_png(out, '维护间隔段下降速率汇总表', OUT_DIR / '表2_维护间隔段下降速率汇总表.png', fontsize=10)


def build_monthly_summary() -> None:
    path = TABLE_DIR / '月度统计表.csv'
    if not path.exists():
        return
    df = pd.read_csv(path)
    keep = [c for c in ['month', 'mean', 'median', 'std', 'count'] if c in df.columns]
    out = format_df(df[keep], digits=4)
    save_table_png(out, '月度透水率统计表', OUT_DIR / '表3_月度透水率统计表.png', fontsize=10)


def build_maintenance_summary() -> None:
    path = TABLE_DIR / '维护前后提升量汇总表.csv'
    if not path.exists():
        return
    df = pd.read_csv(path)
    keep = [c for c in ['维护类型', '有效样本数', '平均提升量', '中位提升量', '标准差', '最小提升量', '最大提升量'] if c in df.columns]
    out = format_df(df[keep], digits=4)
    save_table_png(out, '不同维护类型提升量汇总表', OUT_DIR / '表4_不同维护类型提升量汇总表.png', fontsize=10)


def build_device_feature_summary() -> None:
    path = TABLE_DIR / '设备综合特征表.csv'
    if not path.exists():
        return
    df = pd.read_csv(path)
    keep = [c for c in [
        'device_code', '平均透水率', '变异系数CV', '中维护次数', '大维护次数',
        '平均维护提升量', '维护有效率', '平均区段下降速率(每年)'
    ] if c in df.columns]
    out = format_df(df[keep], digits=4)
    save_table_png(out, '设备综合特征汇总表', OUT_DIR / '表5_设备综合特征汇总表.png', fontsize=9)


def main() -> None:
    configure_chinese_font()
    ensure_dirs()
    build_missing_summary()
    build_decline_summary()
    build_monthly_summary()
    build_maintenance_summary()
    build_device_feature_summary()
    print('已生成 paper_tables 下的论文用表格 PNG。')


if __name__ == '__main__':
    main()
