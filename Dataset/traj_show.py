import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import ScalarFormatter

#資料設定
folder_num_s = 1
folder_num_e = 1
file_num_s = 0
file_num_e = 0
photo_s = 0
photo_e = 5

# 定義檔案路徑
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Cut_Data")
for i in range(folder_num_s,folder_num_e+1):
    folder_path = os.path.join(path, str(i))
    file_name = str(i)
    
    if not os.path.isdir(folder_path):
        continue
    
    for j in range(file_num_s,file_num_e+1):
        file_name2 = file_name + "_" + f"{j:03}"
        
        if not os.path.exists(f"{os.path.join(folder_path, file_name2)}_0000.csv"):
            # print(f"{os.path.join(folder_path, file_name2)}_0000.csv")
            continue

        for k in range(photo_s,photo_e+1):
            file_name3 = file_name2 + "_"  + f"{k:04}"+ ".csv"
            file_path = os.path.join(folder_path, file_name3)
            
            if not os.path.exists(file_path):
                continue
            
            # 讀取 CSV 檔案
            df = pd.read_csv(file_path, header=None, names=['Longitude', 'Latitude', 'Time'])

            # 畫出路徑
            plt.plot(df['Longitude'], df['Latitude'], marker='o', color='b', linestyle='-', markersize=5)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title('Path based on Longitude and Latitude')
            formatter = ScalarFormatter()
            formatter.set_scientific(False) 
            formatter.set_useOffset(False) 
            plt.gca().xaxis.set_major_formatter(formatter)
            plt.gca().yaxis.set_major_formatter(formatter)
            plt.grid(True)
            plt.show()
