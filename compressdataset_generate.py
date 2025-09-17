import os
import subprocess
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import io

def compress_and_get_df(input_csv, threshold="0.0005", exe_path="tdtr.exe"):
    result = subprocess.run(
        [exe_path, input_csv, threshold, "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        print(f"[Error] Compression failed for {input_csv}:\n{result.stderr}")
        return None

    df = pd.read_csv(io.StringIO(result.stdout), header=None, names=["lat", "lon", "time"])
    return df

def create_npz_directly(input_csv, output_npz, threshold="0.0005", exe_path="tdtr.exe"):
    df_original = pd.read_csv(input_csv, header=None, names=["lat", "lon", "time"])
    df_compressed = compress_and_get_df(input_csv, threshold, exe_path)
    if df_compressed is None:
        return False

    original = df_original[["lat", "lon"]].values.astype("float32")
    compressed = df_compressed[["lat", "lon"]].values.astype("float32")

    np.savez(output_npz, original=original, compressed=compressed)
    return True

def process_folders(start_folder, end_folder, cutdata_root, out_root, exe_path, threshold):
    for folder_index in range(start_folder, end_folder + 1):
        folder = str(folder_index)
        input_dir = os.path.join(cutdata_root, folder)
        if not os.path.isdir(input_dir):
            print(f"[Skip] Folder not found: {input_dir}")
            continue

        output_dir = os.path.join(out_root, folder)
        os.makedirs(output_dir, exist_ok=True)

        csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
        if not csv_files:
            print(f"[Info] No CSV files in {input_dir}")
            continue

        print(f"[Folder {folder}] Processing {len(csv_files)} file(s)...")
        for csv_file in tqdm(csv_files, desc=f"Folder {folder}", ncols=80):
            input_csv = os.path.join(input_dir, csv_file)
            output_npz = os.path.join(output_dir, csv_file.replace(".csv", ".npz"))

            success = create_npz_directly(input_csv, output_npz, threshold, exe_path)
            if not success:
                print(f"[Error] Failed to create {output_npz}")

        print(f"[Done] Folder {folder} finished.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress trajectory data into .npz without intermediate .csv")
    parser.add_argument("--threshold", type=str, default="0.0005", help="Compression error threshold")
    parser.add_argument("--start_folder", type=int, default=0, help="Start folder index (inclusive)")
    parser.add_argument("--end_folder", type=int, default=0, help="End folder index (inclusive)")
    parser.add_argument("--exe", type=str, default="tdtr.exe", help="Path to the compression executable")
    parser.add_argument("--cutdata_root", type=str, default="Cut_Data", help="Input root directory")
    parser.add_argument("--out_root", type=str, default="Compressed_Data", help="Output root directory")

    args = parser.parse_args()
    
    process_folders(
        args.start_folder,
        args.end_folder,
        args.cutdata_root,
        args.out_root,
        args.exe,
        args.threshold
    )
