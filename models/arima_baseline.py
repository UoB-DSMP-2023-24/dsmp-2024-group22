import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

from data_preprocess.preprocess import PreProcessLOBData


def retrieve_preprocessed_data(file_path):
    # Create an instance of PreProcessLOBData
    lob_data_processor = PreProcessLOBData(file_path)

    # Process the file to extract and preprocess the data
    lob_data_processor.process_file()

    # Get the preprocessed data
    preprocessed_data = lob_data_processor.preprocess_data()

    return preprocessed_data


def create_time_series(preprocessed_data):
    # Create DataFrame from preprocessed data
    df = pd.DataFrame(preprocessed_data)

    # Extract the time component from the timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s').dt.time

    # Extract bid and ask prices
    bid_prices = df['bids'].apply(lambda x: [bid['price'] for bid in x])
    ask_prices = df['asks'].apply(lambda x: [ask['price'] for ask in x])

    # Take the mean of bid and ask prices to create a single time series
    df['bid_mean'] = bid_prices.apply(lambda x: sum(x) / len(x))
    df['ask_mean'] = ask_prices.apply(lambda x: sum(x) / len(x))

    # Combine bid and ask means
    time_series = df[['timestamp', 'bid_mean', 'ask_mean']]

    return time_series

def arima_forecasting(time_series):
    # Fit ARIMA model
    model = ARIMA(time_series, order=(5, 1, 0))
    model_fit = model.fit()

    # Forecast next 10 steps
    forecast = model_fit.forecast(steps=10)

    return forecast


if __name__ == "__main__":
    # Provide the file path
    file_path = "C:\\Users\\ramkh\\Documents\\Final Year Project\\dsmp-2024-group22\\data_preprocess\\UoB_Set01_2025-01-02LOBs.txt"

    # Retrieve the preprocessed data
    preprocessed_data = retrieve_preprocessed_data(file_path)

    # Create time series from preprocessed data
    time_series = create_time_series(preprocessed_data)

    # Perform ARIMA forecasting
    forecast = arima_forecasting(time_series)

    # Print forecast
    print("ARIMA Forecast:")
    print(forecast)

    # Plot original time series and forecasted values
    plt.plot(time_series.index, time_series.values, label='Original Time Series')
    plt.plot(pd.date_range(start=time_series.index[-1], periods=10, freq='S'), forecast, label='Forecast')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.title('ARIMA Forecasting')
    plt.legend()
    plt.show()
