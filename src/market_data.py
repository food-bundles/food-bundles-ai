"""
Reads all RBA market price xlsx files and returns a human-readable summary
for use as context in the market intelligence AI agent.
"""
import os
import glob
from typing import Optional
import openpyxl
from src.config import config

COMMODITIES = {
    "KURUZA": "Irish Potatoes",
    "INYANYA": "Tomatoes",
    "IBITUNGURU BITUKURA": "Red Onions",
    "INSHYUHYU": "Cabbage",
    "KAROTI": "Carrots",
}

_cache: Optional[str] = None
_cache_mtime: float = 0.0


def _all_files() -> list:
    pattern = os.path.join(config.data_path, "market-prices", "*.xlsx")
    return sorted(glob.glob(pattern))


def _parse_file(filepath: str) -> dict:
    prices = {name: [] for name in COMMODITIES.values()}
    try:
        wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        # Row index 2 (0-based) is the header row with column names
        if len(rows) < 3:
            wb.close()
            return prices
        headers = [str(h).strip() if h else "" for h in rows[2]]
        for row in rows[3:]:
            for col_name, friendly in COMMODITIES.items():
                if col_name in headers:
                    idx = headers.index(col_name)
                    val = row[idx] if idx < len(row) else None
                    if isinstance(val, (int, float)) and val > 0:
                        prices[friendly].append(float(val))
        wb.close()
    except Exception:
        pass
    return prices


def get_market_summary() -> str:
    global _cache, _cache_mtime

    files = _all_files()
    if not files:
        return "No market price data available."

    latest_mtime = max(os.path.getmtime(f) for f in files)
    if _cache and latest_mtime == _cache_mtime:
        return _cache

    # Merge prices across all files for full historical context
    merged = {name: [] for name in COMMODITIES.values()}
    for filepath in files:
        file_prices = _parse_file(filepath)
        for commodity, vals in file_prices.items():
            merged[commodity].extend(vals)

    lines = [f"RBA market prices across {len(files)} daily reports (RWF/kg):"]
    for commodity, vals in merged.items():
        if not vals:
            continue
        avg = round(sum(vals) / len(vals))
        low = round(min(vals))
        high = round(max(vals))
        lines.append(f"  {commodity}: avg {avg} RWF/kg (range {low}\u2013{high}, n={len(vals)} readings)")

    _cache = "\n".join(lines)
    _cache_mtime = latest_mtime
    return _cache
