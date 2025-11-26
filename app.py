# app.py (single-stroke + model integration)
import os, json, subprocess
from pathlib import Path

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError

# streamlit run app.py

st.set_page_config(page_title="Trajectory Drawer → Model", layout="wide")
st.title("Trajectory Drawer → CSV → Model (image & GIF preview)")

with st.sidebar:
    st.header("Export Settings")
    base_lat = st.number_input("Base latitude", value=39.900000, step=0.000001, format="%.6f")
    base_lon = st.number_input("Base longitude", value=116.420000, step=0.000001, format="%.6f")
    deg_per_100px = st.number_input("Degrees per 100 px", value=0.002, step=0.0001, format="%.4f")
    sec_step = st.number_input("Seconds step", min_value=1, value=1, step=1)
    decimals = st.number_input("Decimal places (lat/lon)", min_value=0, max_value=10, value=6, step=1)
    thresh = st.number_input("Down-sample threshold (px)", min_value=0, value=2, step=1)

    st.header("Canvas")
    c_w = st.number_input("Canvas width (px)", min_value=200, value=640, step=10)
    c_h = st.number_input("Canvas height (px)", min_value=200, value=400, step=10)
    stroke_color = st.color_picker("Stroke color", "#0ea5e9")
    stroke_width = st.slider("Stroke width", 1, 12, 3)

    st.header("Model Settings")
    project_root = st.text_input("Your project root (where test.py is)", value=".")
    test_py = st.text_input("test.py path (relative to project root)", value="test_patched.py")
    csv_out_rel = st.text_input("CSV save path (relative to project root)", value="data/test/test.csv")
    out_png_rel = st.text_input("Expected PNG output", value="result/test_kept.png")
    out_gif_rel = st.text_input("Expected GIF output", value="result/test_kept_anim.gif")
    simp_png_rel = st.text_input("Simplified-only PNG", value="result/test_simplified.png")
    cmp_png_rel  = st.text_input("Compare PNG (orig vs simplified)", value="result/test_compare.png")


if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas_0"
if "init_json" not in st.session_state:
    st.session_state.init_json = None

def extract_points_from_fabric_path(path_list, width=None, height=None):
    """Extract (x,y) points from Fabric.js path commands and clip to canvas bounds."""
    pts = []
    for seg in path_list or []:
        if not seg:
            continue
        cmd = seg[0]
        if cmd in ("M", "L"):
            if len(seg) >= 3:
                x, y = float(seg[1]), float(seg[2])
        elif cmd in ("Q", "C"):
            x, y = float(seg[-2]), float(seg[-1])
        else:
            continue

        # ---- 新增：限制筆畫在畫布範圍內 ----
        if width is not None and height is not None:
            x = max(0, min(width, x))
            y = max(0, min(height, y))
        pts.append((x, y))
    return pts


def downsample(points_xy, thresh_px=2.0):
    if not points_xy:
        return []
    out = [points_xy[0]]
    tx, ty = points_xy[0]
    for (x, y) in points_xy[1:]:
        dx = x - tx; dy = y - ty
        if dx*dx + dy*dy >= thresh_px*thresh_px:
            out.append((x, y)); tx, ty = x, y
    return out

def px_to_geo(points_xy, width, height, base_lat, base_lon, deg_per_100px, sec_step, decimals):
    rows = []
    if not points_xy:
        return rows
    deg_per_px = deg_per_100px / 100.0
    ox, oy = width/2.0, height/2.0
    for i, (x, y) in enumerate(points_xy):
        dlon = (x - ox) * deg_per_px
        dlat = -(y - oy) * deg_per_px
        lat = base_lat + dlat
        lon = base_lon + dlon
        rows.append({"lat": f"{lat:.{decimals}f}", "lon": f"{lon:.{decimals}f}", "seconds": i * int(sec_step)})
    return rows

left, right = st.columns(2, gap="large")

with left:
    st.subheader("Draw trajectory (single stroke, new stroke replaces previous)")
    canvas = st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#ffffff",
        height=int(c_h),
        width=int(c_w),
        drawing_mode="freedraw",
        key=st.session_state.canvas_key,
        initial_drawing=st.session_state.init_json,
        update_streamlit=True,
        display_toolbar=True,
    )

    rows = []
    if canvas.json_data is not None and "objects" in canvas.json_data:
        objs = canvas.json_data["objects"]
        if len(objs) >= 1:
            last = objs[-1]
            if len(objs) > 1:
                new_json = {"version": canvas.json_data.get("version", "4.4.0"),
                            "objects": [last]}
                st.session_state.init_json = new_json
                st.session_state.canvas_key = f"canvas_{int(st.session_state.canvas_key.split('_')[-1]) + 1}"
                st.experimental_rerun()
            if last.get("type") == "path":
                pts_xy = extract_points_from_fabric_path(last.get("path", []), width=c_w, height=c_h)
                pts_xy = downsample(pts_xy, thresh_px=float(thresh))
                rows = px_to_geo(pts_xy, c_w, c_h, base_lat, base_lon, deg_per_100px, sec_step, decimals)

    if rows:
        df = pd.DataFrame(rows)
        st.markdown(f"**Preview (first 50 rows) — 原始點數：{len(rows)}**")
        st.dataframe(df.head(50))
        csv_str = "lat,lon,seconds\n" + "\n".join([f"{r['lat']},{r['lon']},{r['seconds']}" for r in rows]) + "\n"
        st.download_button("Download CSV", data=csv_str.encode("utf-8"),
                           file_name="drawn_trajectory.csv", mime="text/csv")
    else:
        st.info("Draw one stroke to preview & enable export. Drawing again will replace the previous stroke.")

    st.markdown("---")
    run_clicked = st.button("✅ 確定並送入模型 (Run test.py)", type="primary", disabled=(len(rows) == 0))

    if run_clicked and rows:
        pr = Path(project_root)
        csv_path = pr / csv_out_rel
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        st.write(f"Saved CSV to: `{csv_path}`")

        test_path = pr / test_py
        if not test_path.exists():
            st.error(f"找不到 test.py：{test_path}"); st.stop()

        st.write("Running model...")
        try:
            proc = subprocess.run(
                ["python", str(test_path)],
                cwd=str(pr),
            )
            st.code(proc.stdout or "", language="bash")
            if proc.returncode != 0:
                st.error("模型執行發生錯誤，stderr 如下：")
                st.code(proc.stderr, language="bash")
        except Exception as e:
            st.exception(e)

with right:
    st.subheader("Model outputs")

    pr = Path(project_root)
    kept_png = pr / out_png_rel
    kept_gif = pr / out_gif_rel
    simp_png = pr / simp_png_rel
    cmp_png  = pr / cmp_png_rel

    # Controls: refresh + view switch
    cols = st.columns(5)
    with cols[0]:
        do_refresh = st.button("🔄 重新整理輸出")
    with cols[1]:
        b1 = st.button("保留點狀況")
    with cols[2]:
        b2 = st.button("簡化軌跡")
    with cols[3]:
        b3 = st.button("軌跡比較")
    with cols[4]:
        b4 = st.button("模型挑選狀況")

    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "kept"
    if b1: st.session_state.view_mode = "kept"
    if b2: st.session_state.view_mode = "simp"
    if b3: st.session_state.view_mode = "cmp"
    if b4: st.session_state.view_mode = "gif"

    # 簡化後點數（讀取 result/test_kept.csv）
    kept_csv = pr / "result/test_kept.csv"
    if kept_csv.exists():
        try:
            dfk = pd.read_csv(kept_csv)
            st.caption(f"簡化後點數：{len(dfk)}")
        except Exception as e:
            st.caption(f"簡化後點數讀取失敗：{e}")
    else:
        st.caption("(簡化後點數：尚未產生 result/test_kept.csv)")

    def show_image_safe(path: Path, label: str):
        if not path.exists():
            st.warning(f"{label} 尚未產生：{path}")
            return
        try:
            if path.stat().st_size == 0:
                st.warning(f"{label} 大小為 0：{path}")
                return
        except OSError as e:
            st.warning(f"{label} 無法存取：{e}")
            return
        try:
            st.image(str(path), caption=str(path))
        except Exception as e:
            st.error(f"{label} 顯示失敗：{e}")

    mode = st.session_state.view_mode
    if mode == "kept":
        show_image_safe(kept_png, "保留點狀況 (PNG)")
    elif mode == "simp":
        show_image_safe(simp_png, "簡化軌跡 (PNG)")
    elif mode == "cmp":
        show_image_safe(cmp_png, "軌跡比較 (PNG)")
    elif mode == "gif":
        show_image_safe(kept_gif, "模型挑選狀況 (GIF)")



