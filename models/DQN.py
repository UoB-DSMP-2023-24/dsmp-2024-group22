import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
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


def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def create_model(n_steps, n_features):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_steps, n_features)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def train_model(model, X_train, y_train, epochs=50, batch_size=128):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    return mse, y_pred


if __name__ == "__main__":
    # Provide the file path
    file_path = "C:\\Users\\ramkh\\Documents\\Final Year Project\\dsmp-2024-group22\\data_preprocess\\UoB_Set01_2025-01-02LOBs.txt"

    # Retrieve the preprocessed data
    preprocessed_data = retrieve_preprocessed_data(file_path)

    # Create time series from preprocessed data
    time_series = create_time_series(preprocessed_data)

    # Define number of time steps
    n_steps = 10

    # Split the data into input (X) and output (y) sequences
    X, y = split_sequence(time_series.values, n_steps)

    # Reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # Define train-test split
    train_size = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    # Create and train the model
    model = create_model(n_steps, n_features)
    model = train_model(model, X_train, y_train)

    # Evaluate the model
    mse, y_pred = evaluate_model(model, X_test, y_test)
    print("Mean Squared Error:", mse)

    # Plot actual vs. predicted values
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('DQN Forecasting')
    plt.legend()
    plt.show()
