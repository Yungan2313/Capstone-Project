import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import ScalarFormatter

folder_path = os.getcwd()

def plot_path(file_name, color, label):
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path, header=None, names=['Longitude', 'Latitude', 'Time'])
    
    plt.plot(df['Longitude'], df['Latitude'], marker='o', linestyle='-', markersize=5, color=color, alpha=0.7, label=label)
    plt.plot(df['Longitude'].iloc[0], df['Latitude'].iloc[0], 'go', markersize=10, label=f'{label} Start')
    plt.plot(df['Longitude'].iloc[-1], df['Latitude'].iloc[-1], 'ro', markersize=10, label=f'{label} End')
    print(f"{file_name} 資料點數:", len(df))

plt.figure(figsize=(10, 8))

# 繪製兩條路徑
plot_path("compress.csv", 'b', 'Compress Path')
plot_path("Uncompress.csv", 'r', 'Uncompress Path')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Compress vs Uncompress Path')

formatter = ScalarFormatter()
formatter.set_scientific(False)
formatter.set_useOffset(False)
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)

plt.grid(True)
plt.legend()
plt.show()