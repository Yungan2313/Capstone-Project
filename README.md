# Capstone-Project: Autoregressive Trajectory Simplification

本專案實作並研究 **自回歸式軌跡簡化模型 (Autoregressive Trajectory Simplification Model)**，基於 Transformer 架構，利用 Encoder-Decoder 機制自動選取並保留關鍵軌跡點，達成資料壓縮的同時保留軌跡結構特徵。

📂 GitHub Repo: [Capstone-Project](https://github.com/Yungan2313/Capstone-Project)

---

## ✨ 功能特點
- **自回歸 Transformer 架構**：Encoder-Decoder 逐步重建軌跡
- **Grid-based Embedding**：將軌跡轉換為相對座標並映射到二維網格
- **Differentiable Top-K**：可微化的選點策略，避免梯度消失
- **Bin Balance (分段平均化 Loss)**：防止點位集中於單一區段
- **Encoder Loss (EL)**：增強 Encoder 選點能力
- 支援 **Geolife dataset** 與自訂資料輸入

---

## 📦 環境需求
- Python 3.8+
- 建議使用 GPU (CUDA 11+)

請先安裝必要套件(requirements.txt)

Capstone-Project/
│── config/                # 模型與實驗參數設定 (YAML)
│── data/                  # 資料集處理 (Geolife步行資料)
│── model/                 # 模型架構 (Transformer, Embedding, Compressor/Decoder)
│── helper/                # 公用工具 (grid utils, metrics)
│── checkpoints/           # 訓練權重儲存
│── train.py               # 訓練入口
│── test.py                # 單一資料測試入口
│── evaluate.py            # 測試資料集測試入口
│── evaluate.py            # 測試/評估入口

## 🚀 使用方式
1. 訓練模型
   python -m train.py 
2. 模型評估
    python evaluate.py --ckpt .\checkpoint\20250910-184624_v2withsmalldecoder\best.pt --out_root .\result\eval --max_iter 200 --patience 10 --cratio 0.2 --save_gif --save_html


