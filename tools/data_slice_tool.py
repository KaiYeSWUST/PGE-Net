import os
import cv2
import numpy as np
import tifffile
from tqdm import tqdm

# 输入：输入数据路径
SOURCE_ROOT = r""
# 输出：模型训练用的最终切片路径
TARGET_ROOT = r""
SUB_SETS = ["train", "val"]

# 切片参数
PATCH_SIZE = 512
OVERLAP_RATE = 0.5
STRIDE = int(PATCH_SIZE * (1 - OVERLAP_RATE))

# 忽略索引 (在该值区域不训练)
IGNORE_INDEX = 255


def make_dirs():
    for subset in SUB_SETS:
        os.makedirs(os.path.join(TARGET_ROOT, "images", subset), exist_ok=True)
        os.makedirs(os.path.join(TARGET_ROOT, "masks", subset), exist_ok=True)
        os.makedirs(os.path.join(TARGET_ROOT, "edges", subset), exist_ok=True)


def normalize_to_uint8(img_data):
    if img_data.dtype == np.uint8:
        return img_data

    norm_img = np.zeros(img_data.shape, dtype=np.uint8)

    if img_data.ndim == 3:
        channels = img_data.shape[2]
        for i in range(channels):
            c = img_data[:, :, i]
            c_min, c_max = c.min(), c.max()
            if c_max > c_min:
                norm_img[:, :, i] = ((c - c_min) / (c_max - c_min) * 255).astype(np.uint8)
            else:
                norm_img[:, :, i] = 0
    else:
        c_min, c_max = img_data.min(), img_data.max()
        if c_max > c_min:
            norm_img = ((img_data - c_min) / (c_max - c_min) * 255).astype(np.uint8)

    return norm_img


def slice_padding_data():
    make_dirs()
    print("开始执行切片 (Source: Padding_gf2 -> Target: gf2)...")

    for subset in SUB_SETS:
        src_img_dir = os.path.join(SOURCE_ROOT, "images", subset)
        src_mask_dir = os.path.join(SOURCE_ROOT, "masks", subset)
        src_edge_dir = os.path.join(SOURCE_ROOT, "edges", subset)

        target_img_dir = os.path.join(TARGET_ROOT, "images", subset)
        target_mask_dir = os.path.join(TARGET_ROOT, "masks", subset)
        target_edge_dir = os.path.join(TARGET_ROOT, "edges", subset)

        if not os.path.exists(src_img_dir): continue

        file_list = [f for f in os.listdir(src_img_dir) if f.endswith(('.tif', '.tiff'))]

        for fname in tqdm(file_list, desc=f"Slicing {subset}"):
            file_id = os.path.splitext(fname)[0]

            try:
                img = tifffile.imread(os.path.join(src_img_dir, fname))
            except Exception as e:
                print(f"读取影像失败: {fname}")
                continue

            if img.ndim == 3 and img.shape[0] <= 4 and img.shape[1] > 32:
                img = np.transpose(img, (1, 2, 0))
            if img.ndim == 2:
                img = img[:, :, np.newaxis]

            img = normalize_to_uint8(img)
            h, w = img.shape[:2]

            mask_path = os.path.join(src_mask_dir, fname)
            edge_path = os.path.join(src_edge_dir, fname)

            try:
                mask = tifffile.imread(mask_path)
                edge = tifffile.imread(edge_path)
            except:
                print(f"对应的 Mask/Edge 读取失败: {fname}, 跳过。")
                continue

            if mask.ndim == 3: mask = mask[0] if mask.shape[0] == 1 else mask[:, :, 0]  # 简单的维度处理
            if edge.ndim == 3: edge = edge[0] if edge.shape[0] == 1 else edge[:, :, 0]

            count_saved = 0
            for y in range(0, h, STRIDE):
                for x in range(0, w, STRIDE):
                    y2, x2 = min(y + PATCH_SIZE, h), min(x + PATCH_SIZE, w)

                    crop_img = img[y:y2, x:x2]
                    crop_mask = mask[y:y2, x:x2]
                    crop_edge = edge[y:y2, x:x2]

                    pad_h = PATCH_SIZE - (y2 - y)
                    pad_w = PATCH_SIZE - (x2 - x)

                    if pad_h > 0 or pad_w > 0:
                        crop_img = cv2.copyMakeBorder(crop_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT,
                                                      value=(0, 0, 0, 0))
                        crop_mask = cv2.copyMakeBorder(crop_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT,
                                                       value=IGNORE_INDEX)
                        crop_edge = cv2.copyMakeBorder(crop_edge, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

                    if np.all(crop_mask == IGNORE_INDEX):
                        continue

                    check_c = min(3, crop_img.shape[2])
                    if np.max(crop_img[:, :, :check_c]) == 0:
                        continue

                    save_name_base = f"{file_id}_{y}_{x}"

                    save_img_path = os.path.join(target_img_dir, save_name_base + ".tif")
                    tifffile.imwrite(save_img_path, crop_img, photometric='rgb')

                    cv2.imwrite(os.path.join(target_mask_dir, save_name_base + ".png"), crop_mask)
                    cv2.imwrite(os.path.join(target_edge_dir, save_name_base + ".png"), crop_edge)

                    count_saved += 1


    print(f"\n所有切片完成！数据位于: {TARGET_ROOT}")

if __name__ == "__main__":
    slice_padding_data()
