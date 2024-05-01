import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mplfinance as mpf
from data_preprocess import PreProcessLOBData


def retrieve_preprocessed_data(file_path):
    # Create an instance of PreProcessLOBData
    lob_data_processor = PreProcessLOBData(file_path)

    # Process the file to extract and preprocess the data
    lob_data_processor.process_file()

    # Get the preprocessed data
    preprocessed_data = lob_data_processor.preprocess_data()

    return preprocessed_data


def calculate_spread(entry):
    best_bid = max(entry['bids'], key=lambda x: x['price'])
    best_ask = min(entry['asks'], key=lambda x: x['price'])
    spread = best_ask['price'] - best_bid['price']
    return spread


def calculate_bid_ask_imbalance(entry):
    total_bid_quantity = sum(bid['quantity'] for bid in entry['bids'])
    total_ask_quantity = sum(ask['quantity'] for ask in entry['asks'])
    imbalance = total_bid_quantity - total_ask_quantity
    return imbalance


if __name__ == "__main__":
    # Provide the file path
    file_path = "C:\\Users\\ramkh\\Documents\\Final Year Project\\dsmp-2024-group22\\data_preprocess\\UoB_Set01_2025-01-02LOBs.txt"

    # Retrieve the preprocessed data
    preprocessed_data = retrieve_preprocessed_data(file_path)

    # Analysis: Calculate spread and bid-ask imbalance
    timestamps = [entry['timestamp'] for entry in preprocessed_data]
    spreads = [calculate_spread(entry) for entry in preprocessed_data]
    bid_ask_imbalances = [calculate_bid_ask_imbalance(entry) for entry in preprocessed_data]

    # Visualize spread over time
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, spreads, color='blue')
    plt.title('Spread Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Spread')
    plt.grid(True)
    plt.show()

    # Visualize bid-ask imbalance over time
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, bid_ask_imbalances, color='red')
    plt.title('Bid-Ask Imbalance Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Imbalance')
    plt.grid(True)
    plt.show()

    # Additional Analyses:

    # 1. Volume Analysis
    total_bid_volume = [sum(bid['quantity'] for bid in entry['bids']) for entry in preprocessed_data]
    total_ask_volume = [sum(ask['quantity'] for ask in entry['asks']) for entry in preprocessed_data]

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, total_bid_volume, color='green', label='Total Bid Volume')
    plt.plot(timestamps, total_ask_volume, color='orange', label='Total Ask Volume')
    plt.title('Volume Analysis Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. Price Movement Analysis
    price_movements = [entry['bids'][0]['price'] - preprocessed_data[i - 1]['bids'][0]['price'] for i, entry in
                       enumerate(preprocessed_data) if i > 0]

    plt.figure(figsize=(10, 5))
    plt.hist(price_movements, bins=20, color='purple', alpha=0.7)
    plt.title('Price Movement Distribution')
    plt.xlabel('Price Movement')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # 3. Market Depth Analysis
    cumulative_bid_volume = [sum(bid['quantity'] for bid in entry['bids'][:n + 1]) for entry in preprocessed_data for n
                             in range(len(entry['bids']))]
    cumulative_ask_volume = [sum(ask['quantity'] for ask in entry['asks'][:n + 1]) for entry in preprocessed_data for n
                             in range(len(entry['asks']))]

    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_bid_volume, label='Cumulative Bid Volume', color='blue')
    plt.plot(cumulative_ask_volume, label='Cumulative Ask Volume', color='red')
    plt.title('Market Depth Analysis')
    plt.xlabel('Price Level')
    plt.ylabel('Cumulative Volume')
    plt.legend()
    plt.grid(True)
    plt.show()


    # Candlestick Chart
    ohlc_data = [(entry['timestamp'], entry['bids'][0]['price'], entry['bids'][-1]['price'], entry['asks'][0]['price'],
                  entry['asks'][-1]['price']) for entry in preprocessed_data]
    df = pd.DataFrame(ohlc_data, columns=['Date', 'Open', 'High', 'Low', 'Close'])
    df['Date'] = pd.to_datetime(df['Date'], unit='s')

    mpf.plot(df.set_index('Date'), type='candle', style='charles', title='Candlestick Chart', ylabel='Price',
             ylabel_lower='Volume')
