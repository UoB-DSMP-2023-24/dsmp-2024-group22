from data_preprocess import retrieve_data

class PreProcessLOBData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.lob_data = []

    def process_file(self):
        lob_processor = retrieve_data.Retrieve_LOB_Data(self.file_path)
        lob_processor.process_file()
        self.lob_data = lob_processor.lob_data

    def preprocess_data(self):
        preprocessed_data = []
        for entry in self.lob_data:
            bids = entry[2][0][1]
            asks = entry[2][1][1]
            if bids and asks:
                bids_dict = [{'price': bid[0], 'quantity': bid[1]} for bid in bids]
                asks_dict = [{'price': ask[0], 'quantity': ask[1]} for ask in asks]
                entry_without_exchange = {'timestamp': entry[0], 'bids': bids_dict, 'asks': asks_dict}
                preprocessed_data.append(entry_without_exchange)
        return preprocessed_data

if __name__ == "__main__":
    file_path = "C:\\Users\\ramkh\\Documents\\Final Year Project\\dsmp-2024-group22\\data_preprocess\\UoB_Set01_2025-01-02LOBs.txt"
    lob_data_processor = PreProcessLOBData(file_path)
    lob_data_processor.process_file()
    preprocessed_data = lob_data_processor.preprocess_data()

    # Display preprocessed data
    for entry in preprocessed_data:
        print(entry)
