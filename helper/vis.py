import os
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point

def graph_show(TIME_STAMP=False, annot_interval=1, use_map=True):
    # TIME_STAMP: whether to annotate timestamps
    # annot_interval: annotate every N points
    # use_map: whether to show background map
    DIR = os.path.join(os.getcwd(), 'datasets/test')
    df = pd.read_csv(os.path.join(DIR, 'test.csv'))

    # Create GeoDataFrame and project
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df.lon, df.lat)],
        crs='EPSG:4326'
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(8, 8))
    gdf.plot(ax=ax, marker='o', markersize=5, color='red', label='trajectory')

    # Add background map if requested
    if use_map:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Mark start and end points
    start = gdf.iloc[0]
    end   = gdf.iloc[-1]
    ax.scatter(start.geometry.x, start.geometry.y,
               marker='o', s=50, color='green', label='start')
    ax.annotate('0',
                xy=(start.geometry.x, start.geometry.y),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, fontweight='bold', color='green')
    ax.scatter(end.geometry.x, end.geometry.y,
               marker='x', s=60, color='blue', label='end')
    ax.annotate(str(end['seconds']),
                xy=(end.geometry.x, end.geometry.y),
                xytext=(5, -10), textcoords='offset points',
                fontsize=8, fontweight='bold', color='red')

    # Annotate timestamps at intervals
    if TIME_STAMP:
        for i, (_, r) in enumerate(gdf.iterrows()):
            if i % annot_interval == 0 and i not in (0, len(gdf)-1):
                ax.annotate(str(r['seconds']),
                            xy=(r.geometry.x, r.geometry.y),
                            xytext=(3, 3), textcoords='offset points',
                            fontsize=6)

    ax.set_axis_off()
    plt.legend()
    plt.tight_layout()
    plt.show()

def data_statistics():
    # 1. 資料夾路徑
    DIR = os.path.join(os.getcwd(), "walk_traj")
    if not os.path.isdir(DIR):
        raise RuntimeError(f"找不到資料夾：{DIR}")

    # 2. 掃所有 CSV
    csv_files = sorted([f for f in os.listdir(DIR) if f.lower().endswith(".csv")])
    num_files = len(csv_files)

    # 3. 統計總筆數
    total_points = 0
    for fn in csv_files:
        df = pd.read_csv(os.path.join(DIR, fn))
        total_points += len(df)

    print(f"共找到 {num_files} 個檔案，總資料點數：{total_points}")

if __name__ == "__main__":
    # data_statistics()
    graph_show(TIME_STAMP = True, annot_interval=5, use_map=True)