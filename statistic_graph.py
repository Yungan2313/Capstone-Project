import matplotlib.pyplot as plt
import numpy as np

# 版本名稱
versions = ["B", "B+DT", "B+BB", "B+BB+DT", "B+EL", "B+EL+DT", "B+EL+BB",  "B+EL+w/oDec", "B+EL+w/oTrDec", "B+EL+w/oTeDec"]

# 勝率數據
wins = [11, 6, 3, 15, 13, 1, 10, 84, 84, 0]
total = 227
win_rates = [w / total * 100 for w in wins]

# 誤差數據
ped_means = [
    24.300259, 27.882390, 15.320585, 13.037474, 
    14.528574, 26.662553, 15.531608, 6.065568, 
    5.557839, 6.065568
]
sed_means = [
    138.839631, 83.250443, 103.792673, 72.865373, 
    49.550662, 72.416316, 51.791953, 55.304078, 
    30.678262, 55.304078
]


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
