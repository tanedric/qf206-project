import os
import sys
from datetime import date, timedelta

try:
    from massive import RESTClient
    CLIENT_NAME = "massive"
except ImportError:
    try:
        from polygon import RESTClient
        CLIENT_NAME = "polygon"
    except ImportError as exc:
        raise SystemExit(
            "Neither 'massive' nor legacy 'polygon-api-client' is installed.\n"
            "Install one with:\n"
            "  python -m pip install massive\n"
            "or:\n"
            "  python -m pip install polygon-api-client"
        ) from exc


def get_api_key() -> str:
    api_key = "lIkF8EGitOC7684WjbEHDDJayCmp_pMg"#os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set MASSIVE_API_KEY or POLYGON_API_KEY before running this script."
        )
    return api_key


def get_date_window() -> tuple[str, str]:
    start_date = os.getenv("MASSIVE_START_DATE") or os.getenv("POLYGON_START_DATE")
    end_date = os.getenv("MASSIVE_END_DATE") or os.getenv("POLYGON_END_DATE")
    if start_date and end_date:
        return start_date, end_date

    # Use a recent window by default so basic plans can usually access the data.
    default_end = date.today() - timedelta(days=1)
    default_start = default_end - timedelta(days=100)
    return default_start.isoformat(), default_end.isoformat()


def main() -> int:
    client = RESTClient(get_api_key())
    symbol = os.getenv("MASSIVE_SYMBOL") or os.getenv("POLYGON_SYMBOL") or "AAPL"
    start_date, end_date = get_date_window()

    aggs = list(
        client.list_aggs(
            symbol,
            1,
            "day",
            start_date,
            end_date,
            limit=50,
        )
    )

    print(f"Client import: {CLIENT_NAME}")
    print(f"Symbol: {symbol}")
    print(f"Date window: {start_date} -> {end_date}")
    print(f"Bars returned: {len(aggs)}")
    if aggs:
        print("First bar:", aggs[0])
        print("Last bar:", aggs[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
