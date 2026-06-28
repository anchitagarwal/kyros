#!/usr/bin/env python3
"""
Download NQ (Nasdaq-100 E-mini) 1-minute historical data from Interactive Brokers
and stitch it into a continuous front-month series.

NQ is a quarterly contract (Mar/Jun/Sep/Dec), so no single contract spans 2024->now.
A continuous future (ContFuture) can't be requested with an endDateTime, so instead
we download each quarterly contract individually (regular Future requests DO accept
endDateTime) and merge them: for any given minute, we keep the bar from the contract
with the nearest expiry, i.e. the true front month. That yields a continuous series
that rolls at each contract's expiry.

To respect IB's historical-data caps we walk each contract backward in chunks and
sleep between requests to stay under the pacing limit (~60 requests / 10 min).

Install:  python3 -m pip install ib-insync
Usage:    python3 download_nq_ib_insync.py [START_DATE] [END_DATE]
          dates are YYYY-MM-DD (default start 2024-01-01, default end = now)
"""

from __future__ import annotations

import asyncio
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ib_insync import IB, Future, util

# --- Connection ---------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 7496          # TWS live. Use 7497 for TWS paper, 4001/4002 for Gateway.
CLIENT_ID = 7

# --- Request shape ------------------------------------------------------------
BAR_SIZE = "1 min"
WHAT_TO_SHOW = "TRADES"
USE_RTH = False                  # False = include overnight/Globex session
CHUNK_DURATION = "5 D"           # window fetched per request (walking backward)
LOOKBACK_DAYS = 100              # how far before expiry each contract is fetched
PACING_SLEEP = 10.0             # seconds between requests (stay under 60/10min)
OUTPUT_FILE = Path("nq_1min_data.csv")

QUARTER_MONTHS = (3, 6, 9, 12)   # NQ expiries: H, M, U, Z


def quarterly_contract_months(start: datetime, end: datetime) -> list[str]:
    """YYYYMM strings for every NQ quarter that could be front-month in [start, end].

    Includes one extra quarter on each side so the boundary contracts are covered.
    """
    months: list[str] = []
    for year in range(start.year - 1, end.year + 2):
        for m in QUARTER_MONTHS:
            months.append(f"{year}{m:02d}")
    return months


async def fetch_contract(
    ib: IB, contract: Future, fetch_lo: datetime, fetch_hi: datetime
) -> list:
    """Walk a single contract backward from fetch_hi to fetch_lo in chunks."""
    collected: list = []
    cursor = fetch_hi
    while cursor > fetch_lo:
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime=cursor,
            durationStr=CHUNK_DURATION,
            barSizeSetting=BAR_SIZE,
            whatToShow=WHAT_TO_SHOW,
            useRTH=USE_RTH,
            formatDate=2,            # UTC
        )
        if not bars:
            break
        collected.extend(bars)
        earliest = _to_utc(bars[0].date)
        print(f"    chunk -> back to {earliest} ({len(bars)} bars)", flush=True)
        if earliest is None or earliest >= cursor:
            break
        cursor = earliest
        await asyncio.sleep(PACING_SLEEP)
    return collected


def write_csv(bars_by_time: dict) -> None:
    """Write the full series to disk, sorted ascending by timestamp."""
    rows = sorted(bars_by_time.values(), key=lambda r: r["date"])
    with OUTPUT_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["date", "open", "high", "low", "close", "volume", "contract"],
        )
        writer.writeheader()
        writer.writerows(rows)


async def download(start: datetime, end: datetime) -> None:
    ib = IB()
    print(f"Connecting to TWS at {HOST}:{PORT} ...")
    await ib.connectAsync(HOST, PORT, clientId=CLIENT_ID)
    print("Connected.")

    # Qualify every quarterly contract so we learn its real expiry date.
    contracts: list[Future] = []
    for ym in quarterly_contract_months(start, end):
        c = Future(symbol="NQ", exchange="CME", currency="USD",
                   lastTradeDateOrContractMonth=ym, includeExpired=True)
        qualified = await ib.qualifyContractsAsync(c)
        if qualified and qualified[0].conId:
            contracts.append(qualified[0])

    # Sort by expiry ascending; front-month-first ordering lets the nearest
    # expiry win when two contracts share a timestamp.
    contracts.sort(key=lambda c: c.lastTradeDateOrContractMonth)

    if not contracts:
        print("No NQ contracts could be qualified. Check market-data permissions.")
        ib.disconnect()
        return

    print(f"Qualified {len(contracts)} contracts: "
          f"{', '.join(c.localSymbol for c in contracts)}", flush=True)

    # timestamp -> row. We process contracts front-first and never overwrite,
    # so each minute keeps the bar from the nearest-expiry (front) contract.
    bars_by_time: dict[str, dict] = {}

    for contract in contracts:
        expiry = _parse_expiry(contract.lastTradeDateOrContractMonth)
        if expiry is None:
            continue
        fetch_hi = min(expiry, end)
        fetch_lo = max(expiry - timedelta(days=LOOKBACK_DAYS), start)
        if fetch_lo >= fetch_hi:
            continue  # contract's front-month window is outside the target range

        print(f"\n{contract.localSymbol}: fetching {fetch_lo.date()} -> "
              f"{fetch_hi.date()}", flush=True)
        bars = await fetch_contract(ib, contract, fetch_lo, fetch_hi)

        added = 0
        for b in bars:
            dt = _to_utc(b.date)
            if dt is None or not (start <= dt <= end):
                continue
            key = dt.strftime("%Y-%m-%d %H:%M:%S")
            if key not in bars_by_time:
                bars_by_time[key] = {
                    "date": key,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                    "contract": contract.localSymbol,
                }
                added += 1
        # Rewrite the (sorted) CSV after every contract so progress is durable.
        if bars_by_time:
            write_csv(bars_by_time)
        print(f"  kept {added} new bars (running total {len(bars_by_time)})",
              flush=True)

    ib.disconnect()

    if not bars_by_time:
        print("\nNo data downloaded.")
        return

    rows = sorted(bars_by_time.values(), key=lambda r: r["date"])
    print(f"\nDone. {len(rows)} bars written to {OUTPUT_FILE}", flush=True)
    print(f"Range: {rows[0]['date']}  ->  {rows[-1]['date']}", flush=True)


def _to_utc(bar_date) -> datetime | None:
    """Normalize an ib_insync bar date (datetime or epoch str) to aware UTC."""
    if isinstance(bar_date, datetime):
        dt = bar_date
    else:
        try:
            dt = datetime.fromtimestamp(int(str(bar_date)), tz=timezone.utc)
        except (ValueError, OverflowError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_expiry(yyyymmdd: str) -> datetime | None:
    try:
        return datetime.strptime(yyyymmdd, "%Y%m%d").replace(
            hour=23, minute=59, second=59, tzinfo=timezone.utc
        )
    except ValueError:
        return None


def _parse_date(s: str, *, end_of_day: bool) -> datetime:
    dt = datetime.strptime(s, "%Y-%m-%d")
    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59)
    return dt.replace(tzinfo=timezone.utc)


if __name__ == "__main__":
    start_arg = sys.argv[1] if len(sys.argv) > 1 else "2024-01-01"
    start_dt = _parse_date(start_arg, end_of_day=False)
    if len(sys.argv) > 2:
        end_dt = _parse_date(sys.argv[2], end_of_day=True)
    else:
        end_dt = datetime.now(timezone.utc)

    print(f"Target range: {start_dt}  ->  {end_dt}")
    util.patchAsyncio()
    asyncio.run(download(start_dt, end_dt))
