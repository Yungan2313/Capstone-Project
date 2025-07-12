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

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config

if __name__ == "__main__":
    config = load_config()

    # 取得參數
    cell_size = config["cell_size"]
    base_lat = config["base_lat"]
    embedding_dim = config["embedding_dim"]