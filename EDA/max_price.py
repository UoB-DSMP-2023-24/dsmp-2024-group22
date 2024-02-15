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
    max_price = df.iloc[:, 1].max()
    max_prices.append(max_price)

print(max_prices)
print(len(max_prices))



max_price_df = pd.DataFrame({'Max Price': max_prices})

output_filepath = r'C:\Users\tlext\Desktop\Group Project\Daily Max Price.csv'
max_price_df.to_csv(output_filepath, index=False)

min_prices = []

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    min_price = df.iloc[:, 1].min()
    min_prices.append(min_price)

print(min_prices)
print(len(min_prices))

min_price_df = pd.DataFrame({'Min Price': min_prices})

output_filepath = r'C:\Users\tlext\Desktop\Group Project\Daily Min Price.csv'
min_price_df.to_csv(output_filepath, index=False)


max_min_df = pd.DataFrame({'Max Price': max_prices, 'Min Price': min_prices})
output_filepath = r'C:\Users\tlext\Desktop\Group Project\Daily Max-Min Price.csv'
max_min_df.to_csv(output_filepath, index=False)

highest_prices = pd.read_csv(r'C:\Users\tlext\Desktop\Group Project\Daily Max Price.csv')
sns.scatterplot(data=highest_prices)
plt.xlabel('Time')
plt.ylabel('Max Price Sold')
plt.show()


lowest_prices = pd.read_csv(r'C:\Users\tlext\Desktop\Group Project\Daily Min Price.csv')
sns.scatterplot(data=lowest_prices)
plt.xlabel('Time')
plt.ylabel('Min Price Sold')
plt.show()

max_min_prices = pd.read_csv(r'C:\Users\tlext\Desktop\Group Project\Daily Max-Min Price.csv')
sns.scatterplot(data=max_min_prices)
plt.xlabel('Time')
plt.ylabel('Max/Min Price Sold')
plt.yticks(np.arange(80, 420, 20))
plt.title('Max and Min Price Sold')
plt.show()