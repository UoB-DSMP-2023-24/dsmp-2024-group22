import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
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
    try:
        # Create DataFrame from preprocessed data
        df = pd.DataFrame(preprocessed_data)

        # Set timestamp as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)

        # Ensure bid and ask prices are numeric
        df['bids'] = df['bids'].apply(lambda x: [float(bid['price']) for bid in x])
        df['asks'] = df['asks'].apply(lambda x: [float(ask['price']) for ask in x])

        df['bid_mean'] = df['bids'].apply(lambda x: sum(x) / len(x))
        df['ask_mean'] = df['asks'].apply(lambda x: sum(x) / len(x))

        # Combine bid and ask means
        time_series = df[['bid_mean', 'ask_mean']].mean(axis=1)

        return time_series

    except Exception as e:
        print("Error:", e)
        return None

def arima_forecasting(time_series):
    # Fit ARIMA model
    model = ARIMA(time_series, order=(0, 1, 5))
    model_fit = model.fit()

    # Forecast next 10 steps
    forecast = model_fit.forecast(steps=10)

    return forecast

def train_arima_model(train_data):
    # Fit ARIMA model
    model = ARIMA(train_data, order=(0, 1, 5))
    model_fit = model.fit()

    return model_fit


def evaluate_arima_model(model, test_data):
    # Forecast next steps
    forecast = model.forecast(steps=len(test_data))

    # Calculate Mean Squared Error
    mse = mean_squared_error(test_data, forecast)

    return mse, forecast


if __name__ == "__main__":
    # Provide the file path
    file_path = "C:\\Users\\ramkh\\Documents\\Final Year Project\\dsmp-2024-group22\\data_preprocess\\UoB_Set01_2025-01-02LOBs.txt"

    # Retrieve the preprocessed data
    preprocessed_data = retrieve_preprocessed_data(file_path)

    # Create time series from preprocessed data
    time_series = create_time_series(preprocessed_data)

    # Split the data into training and test sets
    train_size = int(len(time_series) * 0.8)  # 80% train, 20% test
    train_data, test_data = time_series[:train_size], time_series[train_size:]

    # Train the ARIMA model
    model = train_arima_model(train_data)

    # Evaluate the ARIMA model
    mse, forecast = evaluate_arima_model(model, test_data)

    # Print evaluation metrics
    print("Mean Squared Error:", mse)

    # Plot original time series and forecasted values
    plt.plot(test_data.index, test_data.values, label='Actual')
    plt.plot(test_data.index, forecast, label='Forecast')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.title('ARIMA Forecasting')
    plt.legend()
    plt.show()

