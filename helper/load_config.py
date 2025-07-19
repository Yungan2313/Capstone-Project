import yaml
import os

def load_config(path="config/config.yaml"):
    """
    載入 YAML 格式的 config 設定檔。

    Args:
        path (str): 設定檔路徑，預設為 config/config.yaml

    Returns:
        dict: 包含所有參數設定的 dictionary
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such file or directory: '{path}'")

    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    return cfg

if __name__ == "__main__":
    cfg = load_config()

    # 取得參數
    num_x = cfg["data"]["num_cells_x"]  # 或 cfg["data"]["num_cells_x"]
    dim = cfg["model"]["embedding"]["dim"]
