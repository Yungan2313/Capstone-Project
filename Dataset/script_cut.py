import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm

create_file = True
file_point_count = False
line_max = 0

def parse_plt(file_path):
    """解析 .plt 檔案，跳過標題部分並提取經度、緯度和時間"""
    with open(file_path, 'r') as f:
        # 讀取所有行
        lines = f.readlines()
        start_index = None
        
        # 尋找 "0" 行後的資料行
        for i, line in enumerate(lines):
            if line.strip() == "0":
                start_index = i + 1  # 從 "0" 行的下一行開始讀取
                break
        
        if start_index is None:
            print(f"警告: {file_path} 沒有找到有效的資料行")
            return None
        
        # 只讀取有效資料行並解析
        data = pd.read_csv(file_path, delimiter=",", header=None, skiprows=start_index,
                           names=["Latitude", "Longitude","zero","Altitude", "Days", "Date", "Time"])

    # 提取經度、緯度和時間，並將時間轉為秒
    data = data[["Longitude", "Latitude", "Time"]]  # 只保留經度、緯度和時間
    if create_file == False and file_point_count == True:
        global line_max
        # print(f"data.shape[0]:{data.shape[0]}")
        line_max = max(line_max,data.shape[0])
        
    # 如果時間是以 "hh:mm:ss" 格式表示，將其轉換為總秒數
    def time_to_seconds(time_str):
        try:
            # 假設時間格式是 'hh:mm:ss'
            time_obj = datetime.strptime(time_str, "%H:%M:%S")
            # 返回自午夜起的總秒數
            return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
        except ValueError:
            return None  # 如果時間格式錯誤，返回 None

    # 將時間字串轉換為秒數
    data["Time"] = data["Time"].apply(time_to_seconds)
    
    # 檢查是否成功轉換
    if data["Time"].isnull().any():
        print(f"警告: {file_path} 有時間格式錯誤的資料")

    # 計算時間差並將時間轉為整數 (秒)
    data["Time"] = (data["Time"] - data["Time"].iloc[0]).astype(int)  # 計算時間差，並轉為整數秒
    
    return data

def save_trajectory_chunks(data, output_dir, file_prefix, points_per_chunk,folder_num):
    """每 `points_per_chunk` 個點為一份檔案，並將每份檔案儲存為 CSV"""
    num_points = len(data)
    file_count = 0
    
    # 儲存每 `points_per_chunk` 個點為一份檔案
    for i in range(0, num_points, points_per_chunk):
        chunk_data = data.iloc[i:i + points_per_chunk]
        
        folder_path = os.path.join(output_dir, str(folder_num))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # 若最後一段點數不足，補充至 `points_per_chunk`，最後一個點的時間為最後一個點的時間
        if len(chunk_data) < points_per_chunk:
            last_time = chunk_data["Time"].iloc[-1]
            while len(chunk_data) < points_per_chunk:
                # 使用 concat 替代 append
                chunk_data = pd.concat([chunk_data, chunk_data.iloc[-1:]], ignore_index=True)
                # 使用 loc 修改時間
                chunk_data.loc[chunk_data.index[-1], "Time"] = last_time
        
        # 儲存為新檔案
        output_file = os.path.join(folder_path, f"{folder_num}_{file_prefix}_{file_count:04d}.csv")
        # print(f"儲存檔案: {output_file}")  # 除錯：輸出儲存檔案的路徑
        
        chunk_data[["Longitude", "Latitude", "Time"]].to_csv(output_file, index=False, header=False)
        file_count += 1


def generate_file_list(start_folder, end_folder, root_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")):
    """根據起始和結束資料夾，生成檔案列表"""
    files_to_cut = []
    end_folder = min(end_folder,181)
    # max_ = 0
    for i in tqdm(range(start_folder, end_folder + 1),desc="切割file中"):
        # count = 0
        files_num = []
        folder_path = os.path.join(root_dir, f"{i:03d}", "Trajectory")
        if os.path.isdir(folder_path):
            # 只處理資料夾內的 .plt 檔案
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".plt"):
                    # count+=1
                    files_num.append(os.path.join(folder_path, file_name))
        files_to_cut.append(files_num)
        # print(f"pos {i}:{count}")
        # max_ = max(max_,count)
    # print(f"count:{max_}")
    return files_to_cut

def process_geolife_trajectory(input_dir, output_dir, start_folder, end_folder, points_per_chunk):
    """遍歷指定的資料夾範圍，並處理每個 .plt 檔案"""
    # 確保輸出資料夾存在
    if not os.path.exists(output_dir):
        print(f"創建輸出資料夾: {output_dir}")
        os.makedirs(output_dir)

    # 生成要處理的檔案列表
    files_to_cut = generate_file_list(start_folder, end_folder, input_dir)
    # for f in files_to_cut:
    #     print(f"{0}\n",f)

    for folder_num,folder in enumerate(files_to_cut):
        file_count = 0
        for file in tqdm(folder,desc=f"建立File:{folder_num}資訊"):
            # 讀取並處理該 .plt 檔案
            data = parse_plt(file)
            if create_file:
                # 確認檔案是否成功解析
                if data is not None:
                    # 設定檔案前綴，依照目前檔案數字編號
                    file_prefix = f"{file_count:03d}"
                    
                    # 儲存切割後的檔案
                    save_trajectory_chunks(data, output_dir, file_prefix, points_per_chunk,folder_num)
                    file_count += 1
                else:
                    print(f"跳過檔案: {file}，無法解析")

# 主程式執行
input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")  # Geolife 資料夾路徑
output_dir = "Cut_Data"  # 切割後資料的儲存資料夾

# 設定起始和結束資料夾編號
start_folder = 0  # 起始資料夾編號
end_folder = 1000  # 結束資料夾編號

# 每個檔案的切割點數（例如每 40 個點為一個檔案）
points_per_chunk = 40

process_geolife_trajectory(input_dir, output_dir, start_folder, end_folder, points_per_chunk)
if create_file == False and file_point_count == True:
    print(line_max) 
