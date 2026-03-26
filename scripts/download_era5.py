#!/usr/bin/env python3
"""
Download ERA5 data for monsoon onset benchmarking.

Requirements:
1. Install cdsapi: pip install cdsapi
2. Create CDS account: https://cds.climate.copernicus.eu/
3. Get API key from: https://cds.climate.copernicus.eu/api-how-to
4. Create ~/.cdsapirc with:
   url: https://cds.climate.copernicus.eu/api
   key: YOUR_API_KEY

Usage:
    # Download u-wind for Webster-Yang Index
    python scripts/download_era5.py --years 2019 2020 2021 2022 2023 2024

    # Download precipitation (alternative to IMD data)
    python scripts/download_era5.py --precip --years 2019 2020 2021 2022 2023 2024

    # Download both
    python scripts/download_era5.py --all --years 2019 2020 2021 2022 2023 2024
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Download ERA5 data for monsoon benchmarking"
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
    parser.add_argument(
        "--precip",
        action="store_true",
        help="Download precipitation data (alternative to IMD)",
    )
    parser.add_argument(
        "--winds",
        action="store_true",
        help="Download u-wind data for WYI (default if no flags)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download both precipitation and winds",
    )

    args = parser.parse_args()

    # Default to winds if nothing specified
    if not args.precip and not args.winds and not args.all:
        args.winds = True

    if args.all:
        args.winds = True
        args.precip = True

    # Import here to give helpful error if cdsapi not installed
    try:
        from monsoon_benchmark.data.era5 import (
            download_era5_for_wyi,
            download_era5_precipitation,
        )
    except ImportError as e:
        print(f"Error: {e}")
        print("\nMake sure the package is installed:")
        print("  pip install -e .")
        return 1

    print("ERA5 Download for Monsoon Benchmarking")
    print("=" * 45)
    print(f"Years: {args.years}")
    print(f"Months: {args.months}")
    print(f"Output directory: {args.data_dir}")
    print(f"Download winds (WYI): {args.winds}")
    print(f"Download precipitation: {args.precip}")
    print()

    if args.winds:
        print("=" * 45)
        print("Downloading u-wind data for Webster-Yang Index")
        print("=" * 45)
        download_era5_for_wyi(
            years=args.years,
            data_dir=args.data_dir,
            months=args.months,
        )
        print()

    if args.precip:
        print("=" * 45)
        print("Downloading precipitation data")
        print("=" * 45)
        download_era5_precipitation(
            years=args.years,
            data_dir=args.data_dir,
            months=args.months,
        )
        print()

    print("Download complete!")
    print(f"Files saved to: {Path(args.data_dir).absolute()}")


if __name__ == "__main__":
    main()
