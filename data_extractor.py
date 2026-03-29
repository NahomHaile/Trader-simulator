#!/usr/bin/env python3
"""
NASDAQ Data Extractor & Sample Generator
=========================================
Generates realistic OHLCV (Open, High, Low, Close, Volume) data in CSV format
compatible with the Veteran Trader Agent.

USAGE:
    python data_extractor.py                    # Generate 2 years of sample NASDAQ data
    python data_extractor.py --days 500         # Custom number of trading days
    python data_extractor.py --ticker AAPL      # Custom ticker label
    python data_extractor.py --output my.csv    # Custom output path

DATA FORMAT (what the trading agent expects):
    Date,Open,High,Low,Close,Volume
    2024-01-02,14820.35,14905.12,14780.00,14890.55,4523000000
    ...

You can also feed in real data from any source (Yahoo Finance CSV exports,
Alpha Vantage, Polygon.io, etc.) as long as it has these columns.
"""

import argparse
import csv
import math
import os
import random
from datetime import datetime, timedelta


def generate_nasdaq_data(
    num_days: int = 504,
    start_date: str = "2024-01-02",
    start_price: float = 14800.0,
    ticker: str = "NASDAQ_COMPOSITE",
    output_path: str = "nasdaq_data.csv",
    seed: int = 42,
) -> str:
    """
    Generate realistic NASDAQ composite OHLCV data with:
    - Trend regimes (bull, bear, sideways)
    - Volatility clustering
    - Volume spikes on big moves
    - Gap opens
    - Mean reversion tendencies
    """
    random.seed(seed)
    
    dt = datetime.strptime(start_date, "%Y-%m-%d")
    price = start_price
    base_volume = 4_500_000_000
    volatility = 0.012  # daily vol
    trend = 0.0003      # slight upward drift
    
    rows = []
    regime_counter = 0
    regime_length = random.randint(20, 60)
    regime_type = "bull"  # bull, bear, sideways
    
    for day in range(num_days):
        # Skip weekends
        while dt.weekday() >= 5:
            dt += timedelta(days=1)
        
        # Regime switching (veteran traders recognize these shifts)
        regime_counter += 1
        if regime_counter >= regime_length:
            regime_counter = 0
            regime_length = random.randint(15, 80)
            regime_type = random.choices(
                ["bull", "bear", "sideways"],
                weights=[0.45, 0.25, 0.30],
            )[0]
        
        # Set drift based on regime
        if regime_type == "bull":
            trend = random.gauss(0.0008, 0.0003)
        elif regime_type == "bear":
            trend = random.gauss(-0.0006, 0.0004)
        else:
            trend = random.gauss(0.0, 0.0002)
        
        # Volatility clustering (GARCH-like)
        vol_shock = random.gauss(0, 0.003)
        volatility = max(0.005, min(0.04, 0.95 * volatility + 0.05 * abs(vol_shock) + 0.002))
        
        # Daily return
        daily_return = trend + random.gauss(0, volatility)
        
        # Occasional gap opens (earnings, news)
        gap = 0
        if random.random() < 0.03:
            gap = random.gauss(0, 0.015)
        
        # Price generation
        open_price = price * (1 + gap)
        close_price = open_price * (1 + daily_return)
        
        intraday_range = abs(daily_return) + random.uniform(0.002, 0.01)
        high_price = max(open_price, close_price) * (1 + random.uniform(0, intraday_range * 0.6))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, intraday_range * 0.6))
        
        # Volume (spikes on big moves and regime changes)
        vol_multiplier = 1.0
        if abs(daily_return) > 0.015:
            vol_multiplier = random.uniform(1.5, 2.5)
        elif abs(daily_return) > 0.008:
            vol_multiplier = random.uniform(1.1, 1.6)
        if regime_counter < 3:
            vol_multiplier *= random.uniform(1.2, 1.8)
        
        volume = int(base_volume * vol_multiplier * random.uniform(0.7, 1.3))
        
        rows.append({
            "Date": dt.strftime("%Y-%m-%d"),
            "Open": round(open_price, 2),
            "High": round(high_price, 2),
            "Low": round(low_price, 2),
            "Close": round(close_price, 2),
            "Volume": volume,
        })
        
        price = close_price
        dt += timedelta(days=1)
    
    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "Open", "High", "Low", "Close", "Volume"])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Generated {len(rows)} trading days of data")
    print(f"  Ticker:     {ticker}")
    print(f"  Date range: {rows[0]['Date']} → {rows[-1]['Date']}")
    print(f"  Price range: ${min(r['Low'] for r in rows):,.2f} – ${max(r['High'] for r in rows):,.2f}")
    print(f"  Saved to:   {output_path}")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NASDAQ OHLCV data for the trading agent")
    parser.add_argument("--days", type=int, default=504, help="Number of trading days")
    parser.add_argument("--ticker", type=str, default="NASDAQ_COMPOSITE")
    parser.add_argument("--output", type=str, default="nasdaq_data.csv")
    parser.add_argument("--start-price", type=float, default=14800.0)
    parser.add_argument("--start-date", type=str, default="2024-01-02")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    generate_nasdaq_data(
        num_days=args.days,
        start_date=args.start_date,
        start_price=args.start_price,
        ticker=args.ticker,
        output_path=args.output,
        seed=args.seed,
    )
