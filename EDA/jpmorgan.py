import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

folder_path = r'C:\Users\tlext\Desktop\Group Project'

csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

max_prices = []

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    max_price = df.iloc[:,1].max()
    max_prices.append(max_price)

print(max_prices)
print(len(max_prices))
# dfs = []

# for file in csv_files:
#     file_path = os.path.join(folder_path, file)
#     df = pd.read_csv(file_path, names=['Time', 'Price', 'Quantity'])
#     dfs.append(df)

# merged_dfs = pd.concat(dfs, ignore_index=True)

# print(merged_dfs)

# # output_filepath = r'C:\Users\tlext\Desktop\concattest\Test Tapes merge.csv'
# # merged_dfs.to_csv(output_filepath, index=False)

#new_data = pd.read_csv(r'C:\Users\tlext\Desktop\concattest\Test Tapes merge.csv')
#print(new_data)


# x = new_data['Time']
# y = new_data['Price']

# plt.scatter(x, y)
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.show()

#sns.scatterplot(data=new_data, x='Time', y='Price', hue='Quantity')
#sns.lineplot(data=new_data, x='Time', y='Price')
#plt.show()
