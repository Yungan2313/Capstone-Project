#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從 OpenStreetMap 抓取北京地區的步行路網（walk），
並輸出純 edgelist (s_node e_node) 到 edge.edgelist，
供後續 Node2Vec 等程式使用。
"""
import os
import osmnx as ox
import networkx as nx

def generate_walk_edgelist(place="Beijing, China",
                            output_path="edge.edgelist"):
    """
    Fetch the walk network for a given place and write an edgelist.

    Parameters:
    - place: str, place name recognized by OSMnx (e.g., "Beijing, China").
    - output_path: str, path to write the edgelist file (no header, space-separated).
    """
    # 獲取指定地區的步行路網
    G = ox.graph_from_place(place, network_type="walk")
    # 轉成無向圖以去重 (同時保留所有邊)
    G = G.to_undirected()

    # 確保輸出目錄存在
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 寫成純 edgelist (每行: s_node e_node)
    nx.write_edgelist(G, output_path, data=False)
    num_edges = G.number_of_edges()
    print(f"完成：共匯出 {num_edges} 條邊到 '{output_path}'")

if __name__ == "__main__":
    # 直接執行會生成 edge.edgelist
    generate_walk_edgelist()
