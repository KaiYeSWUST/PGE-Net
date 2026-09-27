import os
import glob
import cv2
import numpy as np

folder_path = r''
paths = glob.glob(os.path.join(folder_path, '*.tif'))

if len(paths) == 0:
    print("没有找到文件，请检查路径。")
else:
    print(f"找到 {len(paths)} 张影像，正在使用 OpenCV 计算...")
    means, stds = [], []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        means.append(img.mean(axis=(0, 1)))
        stds.append(img.std(axis=(0, 1)))

    mean = np.mean(means, axis=0).tolist()
    std = np.mean(stds, axis=0).tolist()

    print("\n========= 计算完成！ =========")
    print(f"mean = {[round(m, 3) for m in mean]},")
    print(f"std  = {[round(s, 3) for s in std]},")
