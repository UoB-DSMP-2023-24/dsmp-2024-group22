import matplotlib.pyplot as plt
import pandas as pd

data1=pd.read_csv("E:/mini project bristol/JPMorgan_Set01/JPMorgan_Set01/LOBs/UoB_Set01_2025-01-02LOBs.txt", sep="[]]]]")
print(data1)
# # 示例数据
# data = [
#     [153.264, 'Exch0',
#      [['bid', [[261, 2], [259, 11], [253, 1], [251, 4], [250, 3], [191, 1], [185, 12], [173, 1], [170, 12]]],
#       ['ask', [[267, 1], [268, 3], [274, 5], [364, 1], [535, 5], [721, 2]]]]],
#     [153.450, 'Exch0',
#      [['bid', [[261, 2], [259, 11], [253, 1], [251, 4], [250, 3], [191, 1], [185, 12], [173, 1], [91, 12]]],
#       ['ask', [[267, 1], [268, 3], [274, 5], [364, 1], [535, 5], [721, 2]]]]],
#     # 添加更多数据点...
# ]
#
# # 初始化列表来存储时间，最高买价和最低卖价
times = []
highest_bids = []
lowest_asks = []

for record in data1:
    time, _, orders = record
    bids, asks = orders[0][1], orders[1][1]

    # 获取最高买价
    highest_bid = max(bids, key=lambda x: x[0])[0] if bids else None
    # 获取最低卖价
    lowest_ask = min(asks, key=lambda x: x[0])[0] if asks else None

    times.append(time)
    highest_bids.append(highest_bid)
    lowest_asks.append(lowest_ask)

# # 绘图
# plt.figure(figsize=(10, 6))
# plt.plot(times, highest_bids, marker='o', linestyle='-', color='green', label='Highest Bid')
# plt.plot(times, lowest_asks, marker='x', linestyle='-', color='red', label='Lowest Ask')
#
# plt.title('Market Bid/Ask Over Time')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Price')
# plt.legend()
# plt.grid(True)
# plt.show()
