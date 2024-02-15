import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

# folder_path = r'C:\Users\tlext\Desktop\Group Project'

# csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# max_prices = []

# for file in csv_files:
#     file_path = os.path.join(folder_path, file)
#     df = pd.read_csv(file_path)
#     max_price = df.iloc[:,1].max()
#     max_prices.append(max_price)

# print(max_prices)
# print(len(max_prices))

# max_price_df = pd.DataFrame({'Max Price': max_prices})

# output_filepath = r'C:\Users\tlext\Desktop\Group Project\Daily Max Price.csv'
# max_price_df.to_csv(output_filepath, index=False)

highest_prices = pd.read_csv(r'C:\Users\tlext\Desktop\Group Project\Daily Max Price.csv')
sns.scatterplot(data=highest_prices)
plt.xlabel('Time')
plt.ylabel('Max Price Sold')
plt.show()
