#!/usr/bin/env python3
"""
Download ERA5 data for Webster-Yang Index computation.

Requirements:
1. Install cdsapi: pip install cdsapi
2. Create CDS account: https://cds.climate.copernicus.eu/
3. Get API key from: https://cds.climate.copernicus.eu/api-how-to
4. Create ~/.cdsapirc with:
   url: https://cds.climate.copernicus.eu/api/v2
   key: YOUR_UID:YOUR_API_KEY

Usage:
    python scripts/download_era5.py --years 2019 2020 2021 2022 2023 2024
    python scripts/download_era5.py --years 2024 --months 5 6 7
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Download ERA5 u-wind data for WYI computation"
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2019, 2020, 2021, 2022, 2023, 2024],
        help="Years to download",
    )
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        default=[4, 5, 6, 7, 8, 9],
        help="Months to download (default: Apr-Sep)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/era5",
        help="Directory to save downloaded files",
    )

    args = parser.parse_args()

    # Import here to give helpful error if cdsapi not installed
    try:
        from monsoon_benchmark.data.era5 import download_era5_for_wyi
    except ImportError as e:
        print(f"Error: {e}")
        print("\nMake sure the package is installed:")
        print("  pip install -e .")
        return 1

    print("ERA5 Download for Webster-Yang Index")
    print("=" * 40)
    print(f"Years: {args.years}")
    print(f"Months: {args.months}")
    print(f"Output directory: {args.data_dir}")
    print()

    download_era5_for_wyi(
        years=args.years,
        data_dir=args.data_dir,
        months=args.months,
    )

    print("\nDownload complete!")
    print(f"Files saved to: {Path(args.data_dir).absolute()}")


if __name__ == "__main__":
    main()
