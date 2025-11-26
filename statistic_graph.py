import matplotlib.pyplot as plt
import numpy as np

# 版本名稱
# versions = ["B", "B+DT", "B+BB", "B+BB+DT", "B+EL", "B+EL+DT", "B+EL+BB",  "B+EL+w/oDec", "B+EL+w/oTrDec", "B+EL+w/oTeDec"]
versions = ["Base Model", "+Bin Balance", "+Encoder Loss", "+BB+EL", "+EL w/o Dec", "+EL w/o TrDec", "+EL w/o TeDec"]
# 勝率數據
# wins = [11, 6, 3, 15, 13, 1, 10, 84, 84, 0]
wins = [11, 18, 13, 11, 87, 87, 0]
total = 227
win_rates = [w / total * 100 for w in wins]

# 誤差數據
# ped_means = [
#     24.300259, 27.882390, 15.320585, 13.037474, 
#     14.528574, 26.662553, 15.531608, 6.065568, 
#     5.557839, 6.065568
# ]
ped_means = [24.300259, 13.037474, 14.528574, 15.531608, 6.065568, 5.557839, 6.065568]
# sed_means = [
#     138.839631, 83.250443, 103.792673, 72.865373, 
#     49.550662, 72.416316, 51.791953, 55.304078, 
#     30.678262, 55.304078
# ]
sed_means = [138.839631, 72.865373, 49.550662, 51.791953, 55.304078, 30.678262, 55.304078]


x = np.arange(len(versions))
width = 0.35

# 圖1: PED & SED
fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, ped_means, width, label='PED_mean')
rects2 = ax.bar(x + width/2, sed_means, width, label='SED_mean')

ax.set_ylabel('Error Value')
ax.set_title('PED & SED Mean Comparison')
ax.set_xticks(x)
ax.set_xticklabels(versions)
ax.legend()
ax.bar_label(rects1, fmt="%.1f", padding=3)
ax.bar_label(rects2, fmt="%.1f", padding=3)
ax.set_xticklabels(versions, rotation=20, ha='right')

plt.tight_layout()
plt.show()

# 圖2: 勝率
fig2, ax2 = plt.subplots(figsize=(8, 5))
bars = ax2.bar(versions, win_rates, color='skyblue')

ax2.set_ylabel('Win Rate (%)')
ax2.set_title('Win Rate Across Versions')
ax2.bar_label(bars, fmt="%.1f%%", padding=3)
ax2.set_xticklabels(versions, rotation=20, ha='right')


plt.tight_layout()
plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # 類別名稱（X軸改為指標）
# categories = ["PED", "SED"]

# # Base Model & Best Model 數據（依照 categories 順序）
# base_values = [24.300259, 138.839631]
# best_values = [5.557839, 30.678262]

# x = np.arange(len(categories))  # [0, 1]
# width = 0.35  # 柱子間距

# fig, ax = plt.subplots(figsize=(10, 4))

# # Base Model 柱子（偏左）
# rects1 = ax.bar(x - width/2, base_values, width, label='Base Model', color='orange')
# # Best Model 柱子（偏右）
# rects2 = ax.bar(x + width/2, best_values, width, label='Best Model', color='tab:blue')

# # 標籤 & 標題
# ax.set_ylabel('Error Value')
# ax.set_title('Base Model vs Best Model on PED & SED')
# ax.set_xticks(x)
# ax.set_xticklabels(categories, fontsize=14)
# ax.legend()

# # 標上數值
# ax.bar_label(rects1, fmt="%.1f", padding=3)
# ax.bar_label(rects2, fmt="%.1f", padding=3)

# plt.tight_layout()
# plt.subplots_adjust(top=0.95)
# plt.show()
