import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from data_preprocess.preprocess import PreProcessLOBData


class DQN:
    def __init__(self, state_shape, num_actions):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=self.state_shape),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.num_actions, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict(self, state):
        return self.model.predict(state)

    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=0)


def prepare_data(preprocessed_data):
    states = []
    targets = []
    for i in range(1, len(preprocessed_data)):
        state = np.array(preprocessed_data[i - 1]['bids'] + preprocessed_data[i - 1]['asks'])
        next_state = np.array(preprocessed_data[i]['bids'] + preprocessed_data[i]['asks'])
        action = np.array(preprocessed_data[i]['asks'][0]['price'] - preprocessed_data[i]['bids'][0]['price'])
        reward = action  # This could be replaced with a more meaningful reward function

        states.append(state)
        targets.append(reward + next_state.max())  # Q-learning target

    return np.array(states), np.array(targets)


if __name__ == "__main__":
    # Provide the file path
    file_path = "C:\\Users\\ramkh\\Documents\\Final Year Project\\dsmp-2024-group22\\data_preprocess\\UoB_Set01_2025-01-02LOBs.txt"

    # Retrieve the preprocessed data using your preprocess class
    lob_data_processor = PreProcessLOBData(file_path)
    lob_data_processor.process_file()
    preprocessed_data = lob_data_processor.preprocess_data()

    # Prepare data for DQN
    states, targets = prepare_data(preprocessed_data)

    # Define DQN model
    state_shape = states.shape[1:]
    num_actions = 1  # Number of actions (one for this simple example)
    dqn = DQN(state_shape, num_actions)

    # Train DQN
    dqn.train(states, targets)

    # Predict with DQN (not meaningful in this example)
    predictions = dqn.predict(states)

    # Print predictions (for demonstration)
    print(predictions)
