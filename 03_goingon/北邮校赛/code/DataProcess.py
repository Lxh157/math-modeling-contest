# cd 03_goingon\北邮校赛\code
# python DataProcess.py
from pathlib import Path

import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = CODE_DIR.parent / "data" / "原始数据"


def sanitize_sheet_name(sheet_name: str) -> str:
    """Make sheet names safe for Windows file names."""
    invalid_chars = '<>:"/\\|?*'
    sanitized = "".join("_" if char in invalid_chars else char for char in sheet_name)
    return sanitized.strip() or "Sheet"


def convert_excel_to_csv(excel_path: Path) -> list[Path]:
    """Convert one Excel workbook to CSV file(s)."""
    excel_file = pd.ExcelFile(excel_path)
    created_files: list[Path] = []

    if len(excel_file.sheet_names) == 1:
        dataframe = pd.read_excel(excel_path, sheet_name=excel_file.sheet_names[0])
        output_path = excel_path.with_suffix(".csv")
        dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
        created_files.append(output_path)
        return created_files

    for sheet_name in excel_file.sheet_names:
        dataframe = pd.read_excel(excel_path, sheet_name=sheet_name)
        output_path = excel_path.with_name(
            f"{excel_path.stem}__{sanitize_sheet_name(sheet_name)}.csv"
        )
        dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
        created_files.append(output_path)

    return created_files


def main() -> None:
    excel_files = sorted(RAW_DATA_DIR.glob("*.xlsx"))
    if not excel_files:
        raise FileNotFoundError(f"未在目录中找到 xlsx 文件：{RAW_DATA_DIR}")

    created_files: list[Path] = []
    for excel_path in excel_files:
        created_files.extend(convert_excel_to_csv(excel_path))

    print("转换完成，生成的 CSV 文件如下：")
    for csv_path in created_files:
        print(csv_path.name)


if __name__ == "__main__":
    main()
