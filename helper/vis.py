import os
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import contextily as ctx
import random
from shapely.geometry import Point

def graph_show_plus(select='specific', filename=None, DIR=None,
                    TIME_STAMP=False, annot_interval=1, use_map=True,
                    show_filename='title', repeat=1, random_unique=True):
    """
    show_filename: 'title' | 'inset' | False/'none'
    """
    if DIR is None:
        DIR = os.path.join(os.getcwd(), 'data/datasets')

    if select not in ('random', 'specific'):
        raise ValueError("select 必須是 'random' 或 'specific'")

    # 內部：畫單一檔案
    def _plot_one(fullpath, fname):
        df = pd.read_csv(fullpath)

        gdf = gpd.GeoDataFrame(
            df,
            geometry=[Point(xy) for xy in zip(df.lon, df.lat)],
            crs='EPSG:4326'
        ).to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=(8, 8))
        gdf.plot(ax=ax, marker='o', markersize=5, color='red', label='trajectory')

        if use_map:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

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

        if TIME_STAMP:
            for i, (_, r) in enumerate(gdf.iterrows()):
                if i % annot_interval == 0 and i not in (0, len(gdf)-1):
                    ax.annotate(str(r['seconds']),
                                xy=(r.geometry.x, r.geometry.y),
                                xytext=(3, 3), textcoords='offset points',
                                fontsize=6)

        # === 在「標題」或「圖內」顯示檔名 ===
        if show_filename in ('title', 'Title', 'TITLE'):
            # 顯示檔名於標題（靠左，與圖稍微拉開距離）
            ax.set_title(fname, loc='left', pad=12, fontsize=12, fontweight='bold')
        elif show_filename in ('inset', True):
            # 舊版的圖內角落做法（保留相容性）
            ax.text(0.01, 0.99, fname,
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white',
                              alpha=0.8, edgecolor='none'))
        else:
            pass  # 不顯示

        ax.set_axis_off()
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close(fig)

        print(f"顯示檔案：{fullpath}")

    shown = []

    if select == 'random':
        csv_files = [f for f in os.listdir(DIR) if f.lower().endswith('.csv')]
        if not csv_files:
            raise RuntimeError(f"資料夾內沒有 CSV 檔：{DIR}")

        if random_unique and repeat <= len(csv_files):
            picks = random.sample(csv_files, repeat)
        else:
            picks = [random.choice(csv_files) for _ in range(repeat)]

        for fn in picks:
            fullpath = os.path.join(DIR, fn)
            _plot_one(fullpath, fn)
            shown.append(fn)

    else:  # specific
        if filename is None:
            filename = '010_000003.csv'

        if isinstance(filename, (list, tuple)):
            picks = list(filename)
            if repeat > 1:
                times = repeat
                seq = []
                while len(seq) < times:
                    need = times - len(seq)
                    seq.extend(picks[:need])
                picks = seq
        else:
            picks = [filename] * repeat

        for fn in picks:
            fullpath = os.path.join(DIR, fn)
            if not os.path.isfile(fullpath):
                print(f"警告：找不到 {fullpath}，略過。")
                continue
            _plot_one(fullpath, fn)
            shown.append(fn)

    return shown
def graph_show(TIME_STAMP=False, annot_interval=1, use_map=True):
    # TIME_STAMP: whether to annotate timestamps
    # annot_interval: annotate every N points
    # use_map: whether to show background map
    # DIR = os.path.join(os.getcwd(), 'data/test')
    DIR = os.path.join(os.getcwd(), 'data/datasets')
    # df = pd.read_csv(os.path.join(DIR, 'test.csv'))
    df = pd.read_csv(os.path.join(DIR, '085_000094.csv'))
    

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
    # 隨機連看 5 張，檔名放在「標題」
    # graph_show_plus(select='random',
    #                 DIR=os.path.join(os.getcwd(), 'data/datasets'),
    #                 TIME_STAMP=False, annot_interval=5, use_map=True,
    #                 repeat=10, show_filename='title')

    # 指定清單，依序畫出各 1 張，檔名放在「標題」
    # graph_show_plus(select='specific',
    #                 filename=['a.csv','b.csv','c.csv'],
    #                 DIR=os.path.join(os.getcwd(), 'data/datasets'),
    #                 show_filename='title', repeat=1)