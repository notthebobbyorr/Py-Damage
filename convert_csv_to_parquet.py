from __future__ import annotations

from pathlib import Path
import pandas as pd


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found.")
        return

    converted = 0
    for csv_path in csv_files:
        parquet_path = csv_path.with_suffix(".parquet")
        df = pd.read_csv(csv_path)
        df.to_parquet(parquet_path, index=False)
        converted += 1
        print(f"Converted {csv_path.name} -> {parquet_path.name}")

    print(f"Done. Converted {converted} files.")


if __name__ == "__main__":
    main()
