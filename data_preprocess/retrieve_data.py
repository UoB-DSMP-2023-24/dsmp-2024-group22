import re

class Retrieve_LOB_Data:
    def __init__(self, file_path):
        self.file_path = file_path
        self.lob_data = []

    def process_file(self):
        with open(self.file_path, "r") as file:
            for line in file:
                line = line.strip()
                line = line[1:-1]
                parts = line.split(", ")

                timestamp = float(parts[0])
                exchange = parts[1]

                bid_match = re.search(r"\['bid', \[\[(.*?)\]\]\]", line)
                ask_match = re.search(r"\['ask', \[\[(.*?)\]\]\]", line)

                bids = [] if not bid_match else [list(map(int, item.split(", "))) for item in bid_match.group(1).split("], [")]
                asks = [] if not ask_match else [list(map(int, item.split(", "))) for item in ask_match.group(1).split("], [")]

                self.lob_data.append([timestamp, exchange, [['bid', bids], ['ask', asks]]])

    def print_data(self, num_elements=20):
        for i in range(min(num_elements, len(self.lob_data))):
            print(self.lob_data[i])

# Example usage:
file_path = "C:\\Users\\ramkh\\Documents\\Final Year Project\\dsmp-2024-group22\\data_preprocess\\UoB_Set01_2025-01-02LOBs.txt"
lob_processor = Retrieve_LOB_Data(file_path)
lob_processor.process_file()
lob_processor.print_data()
