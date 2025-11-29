"""
Download Stock Market Data for BPE Tokenizer Training
Downloads historical stock data from multiple sources and formats it for tokenization
"""

import sys
import io

# Fix console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("Installing required packages...")
import subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "yfinance", "pandas"], check=True)

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def download_stock_data():
    """Download historical stock data for multiple companies"""
    
    # Major stocks from different sectors
    tickers = [
        # Tech
        'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC',
        # Finance
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'MRK',
        # Consumer
        'AMZN', 'WMT', 'HD', 'NKE', 'MCD', 'SBUX',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB',
        # Industrial
        'BA', 'CAT', 'GE', 'MMM',
        # Indices
        '^GSPC', '^DJI', '^IXIC'  # S&P 500, Dow Jones, NASDAQ
    ]
    
    print(f"\nDownloading data for {len(tickers)} stocks...")
    print("This will download 5 years of daily data\n")
    
    # Download 5 years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    all_data = []
    
    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"[{i}/{len(tickers)}] Downloading {ticker}...", end=' ')
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if not df.empty:
                df['Ticker'] = ticker
                all_data.append(df)
                print(f"✓ ({len(df)} days)")
            else:
                print("✗ No data")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Combine all data
    print(f"\nCombining data from {len(all_data)} stocks...")
    combined_df = pd.concat(all_data)
    combined_df = combined_df.reset_index()
    
    print(f"Total records: {len(combined_df):,}")
    
    return combined_df

def format_for_tokenization(df):
    """Format stock data as text for BPE training with labels for better compression"""
    
    print("\nFormatting data for tokenization with labels...")
    
    # Sector mapping for major stocks
    sector_map = {
        'AAPL': 'TECH', 'MSFT': 'TECH', 'GOOGL': 'TECH', 'META': 'TECH', 
        'NVDA': 'TECH', 'TSLA': 'AUTO', 'AMD': 'TECH', 'INTC': 'TECH',
        'JPM': 'FIN', 'BAC': 'FIN', 'WFC': 'FIN', 'GS': 'FIN', 'MS': 'FIN', 'C': 'FIN',
        'JNJ': 'HEALTH', 'UNH': 'HEALTH', 'PFE': 'HEALTH', 'ABBV': 'HEALTH', 
        'TMO': 'HEALTH', 'MRK': 'HEALTH',
        'AMZN': 'RETAIL', 'WMT': 'RETAIL', 'HD': 'RETAIL', 'NKE': 'RETAIL', 
        'MCD': 'RETAIL', 'SBUX': 'RETAIL',
        'XOM': 'ENERGY', 'CVX': 'ENERGY', 'COP': 'ENERGY', 'SLB': 'ENERGY',
        'BA': 'INDUST', 'CAT': 'INDUST', 'GE': 'INDUST', 'MMM': 'INDUST',
        '^GSPC': 'INDEX', '^DJI': 'INDEX', '^IXIC': 'INDEX'
    }
    
    def get_volume_category(volume_millions):
        """Categorize volume for pattern repetition"""
        if volume_millions < 50:
            return 'LOW'
        elif volume_millions < 150:
            return 'MED'
        else:
            return 'HIGH'
    
    def get_price_range(price):
        """Categorize price into ranges"""
        if price < 50:
            return 'UNDER50'
        elif price < 100:
            return 'UNDER100'
        elif price < 200:
            return 'UNDER200'
        elif price < 500:
            return 'UNDER500'
        else:
            return 'OVER500'
    
    lines = []
    for _, row in df.iterrows():
        ticker = row['Ticker']
        sector = sector_map.get(ticker, 'OTHER')
        
        # Round prices to 1 decimal
        open_price = round(row['Open'], 1)
        high_price = round(row['High'], 1)
        low_price = round(row['Low'], 1)
        close_price = round(row['Close'], 1)
        
        # Volume in millions
        volume_millions = round(row['Volume'] / 1_000_000, 1)
        vol_category = get_volume_category(volume_millions)
        
        # Price range
        price_range = get_price_range(close_price)
        
        # Day of week for more repetition
        day_of_week = row['Date'].strftime('%a').upper()  # MON, TUE, WED, etc.
        
        # Format with labels for better compression
        # Pattern: SECTOR|TICKER|YEAR-MONTH|DAY|RANGE|OPEN:X|HIGH:X|LOW:X|CLOSE:X|VOL:CAT
        line = (
            f"{sector}|{ticker}|"
            f"{row['Date'].strftime('%Y-%m')}|"  # Month only
            f"{day_of_week}|"  # Day of week
            f"{price_range}|"
            f"OPEN:{open_price}|"
            f"HIGH:{high_price}|"
            f"LOW:{low_price}|"
            f"CLOSE:{close_price}|"
            f"VOL:{vol_category}"
        )
        lines.append(line)
    
    # Join with newlines
    text = '\n'.join(lines)
    
    return text

def save_corpus(text, filename='stock_corpus.txt'):
    """Save the formatted text corpus"""
    
    print(f"\nSaving to {filename}...")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)
    
    size_mb = len(text) / (1024 * 1024)
    print(f"✓ Saved {len(text):,} characters (~{size_mb:.2f} MB)")
    
    return filename

if __name__ == "__main__":
    print("=" * 70)
    print("Stock Market Data Downloader for BPE Tokenizer")
    print("=" * 70)
    
    # Download data
    df = download_stock_data()
    
    # Format for tokenization
    text = format_for_tokenization(df)
    
    # Save corpus
    filename = save_corpus(text)
    
    print("\n" + "=" * 70)
    print("✓ Download complete!")
    print(f"  Corpus saved to: {filename}")
    print(f"  Total records: {len(df):,}")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    print("\nNext step: Run 'python train_tokenizer.py'")
    print("=" * 70)
