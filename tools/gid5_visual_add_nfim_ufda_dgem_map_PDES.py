import sys
import os

MMSEG_ROOT = r""
if MMSEG_ROOT not in sys.path:
    sys.path.insert(0, MMSEG_ROOT)

import inspect
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from osgeo import gdal
from tqdm import tqdm
from mmseg.apis import init_model
from mmseg.structures import SegDataSample

CONFIG_FILE = (r"")
CHECKPOINT_FILE = (r"")
INPUT_IMG_PATH = (r"")
GT_IMG_PATH = (r"")
OUTPUT_PATH = (r"")
MAPS_DIR = (r"")

os.makedirs(MAPS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

DEVICE = "cuda:0"
NUM_CLASSES = 2
WATER_CLASS_INDEX = 1

CROP_SIZE = 512
STRIDE = 128

def read_geotiff(path):
    dataset = gdal.Open(path)

    if dataset is None:
        raise FileNotFoundError(f"无法打开文件: {path}")

    if dataset.RasterCount < 4:
        raise ValueError(
            f"预期至少包含4个波段，但{path}只有"
            f"{dataset.RasterCount}个波段"
        )

    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    image = np.stack(
        [
            dataset.GetRasterBand(i + 1).ReadAsArray()
            for i in range(4)
        ],
        axis=0
    )

    image = image[[2, 1, 0, 3], :, :]

    dataset = None

    return image, geo_transform, projection

def save_geotiff(
    data,
    path,
    geo_transform,
    projection,
    gdal_type=gdal.GDT_Byte
):
    if data.ndim != 2:
        raise ValueError(
            f"save_geotiff只支持二维单波段数据，收到形状: {data.shape}"
        )

    driver = gdal.GetDriverByName("GTiff")
    rows, cols = data.shape

    dataset = driver.Create(
        path,
        cols,
        rows,
        1,
        gdal_type,
        options=["COMPRESS=LZW"]
    )

    if dataset is None:
        raise RuntimeError(f"无法创建GeoTIFF文件: {path}")

    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(data)
    dataset.GetRasterBand(1).FlushCache()
    dataset.FlushCache()
    dataset = None
    print(f"结果已保存为TIFF: {path}")

def preprocess_image(image):
    processed_image = np.zeros_like(image, dtype=np.float32)
    for channel_index in range(image.shape[0]):
        channel = image[channel_index].astype(np.float32)
        value_min = float(channel.min())
        value_max = float(np.percentile(channel, 99.9))
        clipped = np.clip(channel, value_min, value_max)
        if value_max > value_min:
            processed_image[channel_index] = (
                (clipped - value_min)
                / (value_max - value_min)
                * 255.0
            )
        else:
            processed_image[channel_index] = 0.0
    return processed_image

def get_weight_mask(crop_size):
    window_1d = np.hanning(crop_size)
    window_2d = np.outer(window_1d, window_1d)
    window_2d = np.maximum(window_2d, 1e-4)

    return window_2d.astype(np.float32)

def get_sliding_positions(length, crop_size, stride):
    if length <= crop_size:
        return [0]
    positions = list(range(0, length - crop_size + 1, stride))
    last_position = length - crop_size
    if positions[-1] != last_position:
        positions.append(last_position)
    return positions

def pad_image_if_needed(image, crop_size):
    _, height, width = image.shape

    pad_height = max(crop_size - height, 0)
    pad_width = max(crop_size - width, 0)

    if pad_height == 0 and pad_width == 0:
        return image, height, width
    padded = np.pad(
        image,
        (
            (0, 0),
            (0, pad_height),
            (0, pad_width)
        ),
        mode="reflect"
    )
    return padded, height, width

def create_attention_storage():
    keys = [
        # NFIM
        "nfim1",
        "nfim2",

        # TPUD/UFDA continuous maps
        "ufda_entropy",
        "ufda_spatial",
        "ufda_physical",
        "ufda_joint_base",
        "joint_mask_highres",

        # Semantic and physical evidence
        "water_probability",
        "semantic_water_mask",
        "semantic_background_mask",
        "physical_water_mask",
        "physical_background_mask",

        # Restricted-evolution-region construction
        "high_uncertainty_mask",
        "explicit_conflict_mask",
        "potential_conflict_mask",
        "initial_conflict_mask",
        "reliable_conflict_mask",
        "restricted_evolution_zone",

        # DGEM physical fields
        "dgem_gravity",
        "corrected_gravity_field",
        "checking_nir_base",
        "checking_ndwi",

        # PDE evolution
        "pde_phi_initial",
        "pde_phi_final",
        "pde_phi_delta",
        "pde_diffusion_coeff",
        "pde_gravity_x",
        "pde_gravity_y",
        "pde_gravity_magnitude",
    ]
    return {key: None for key in keys}

def tensor_to_numpy(value):
    if value is None:
        return None

    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()

    if isinstance(value, np.ndarray):
        return value

    return None

def binary_logits_to_log_odds(value):
    array = tensor_to_numpy(value)

    if array is None:
        return None

    if array.ndim != 4 or array.shape[1] != 2:
        raise ValueError(
            "PDE logits必须具有[B, 2, H, W]形状，"
            f"实际收到: {array.shape}"
        )

    return array[:, 1:2, :, :] - array[:, 0:1, :, :]

def register_interception_hooks(model, attention_maps):
    hook_handles = []

    def get_attention_hook(name):
        def hook(module, inputs, output):
            output_array = tensor_to_numpy(output)
            if output_array is not None:
                attention_maps[name] = output_array
        return hook

    # NFIM hooks
    if hasattr(model, "backbone"):
        if (
            hasattr(model.backbone, "nfim1")
            and hasattr(model.backbone.nfim1, "sigmoid")
        ):
            handle = model.backbone.nfim1.sigmoid.register_forward_hook(
                get_attention_hook("nfim1")
            )
            hook_handles.append(handle)
            print("检测到NFIM1模块，已设置特征图拦截")

        if (
            hasattr(model.backbone, "nfim2")
            and hasattr(model.backbone.nfim2, "sigmoid")
        ):
            handle = model.backbone.nfim2.sigmoid.register_forward_hook(
                get_attention_hook("nfim2")
            )
            hook_handles.append(handle)
            print("检测到NFIM2模块，已设置特征图拦截")

    if not hasattr(model, "decode_head"):
        raise AttributeError("当前模型不存在decode_head")

    decode_head = model.decode_head

    if not hasattr(decode_head, "intercepted_vars"):
        raise AttributeError(
            "decode_head不存在intercepted_vars，"
        )

    print("成功定位解码头，正在注入中间变量拦截器")

    original_decode_head_forward = decode_head.forward

    import types

    def intercepted_forward(self, inputs):
        output = original_decode_head_forward(inputs)

        intercepted_vars = getattr(self, "intercepted_vars", {})

        direct_keys = [
            "ufda_entropy",
            "ufda_spatial",
            "ufda_physical",
            "ufda_joint_base",
            "joint_mask_highres",
            "water_probability",
            "semantic_water_mask",
            "semantic_background_mask",
            "physical_water_mask",
            "physical_background_mask",
            "high_uncertainty_mask",
            "explicit_conflict_mask",
            "potential_conflict_mask",
            "initial_conflict_mask",
            "reliable_conflict_mask",
            "restricted_evolution_zone",
            "checking_nir_base",
            "checking_ndwi",
            "corrected_gravity_field",
        ]

        for key in direct_keys:
            if key in intercepted_vars:
                value = tensor_to_numpy(intercepted_vars[key])

                if value is not None:
                    attention_maps[key] = value

        #DGEM physical fields
        if "gravity_field" in intercepted_vars:
            attention_maps["dgem_gravity"] = tensor_to_numpy(
                intercepted_vars["gravity_field"]
            )

        if "corrected_gravity_field" in intercepted_vars:
            attention_maps["corrected_gravity_field"] = tensor_to_numpy(
                intercepted_vars["corrected_gravity_field"]
            )

        if "pde_initial_logits" in intercepted_vars:
            attention_maps["pde_phi_initial"] = binary_logits_to_log_odds(
                intercepted_vars["pde_initial_logits"]
            )

        if "pde_final_logits" in intercepted_vars:
            attention_maps["pde_phi_final"] = binary_logits_to_log_odds(
                intercepted_vars["pde_final_logits"]
            )

        if (
                "pde_initial_logits" in intercepted_vars
                and "pde_final_logits" in intercepted_vars
        ):
            phi_initial = binary_logits_to_log_odds(
                intercepted_vars["pde_initial_logits"]
            )
            phi_final = binary_logits_to_log_odds(
                intercepted_vars["pde_final_logits"]
            )
            attention_maps["pde_phi_delta"] = phi_final - phi_initial

        if "pde_diffusion_coeff" in intercepted_vars:
            attention_maps["pde_diffusion_coeff"] = tensor_to_numpy(
                intercepted_vars["pde_diffusion_coeff"]
            )

        if "pde_gravity_x" in intercepted_vars:
            attention_maps["pde_gravity_x"] = tensor_to_numpy(
                intercepted_vars["pde_gravity_x"]
            )

        if "pde_gravity_y" in intercepted_vars:
            attention_maps["pde_gravity_y"] = tensor_to_numpy(
                intercepted_vars["pde_gravity_y"]
            )

        if (
                "pde_gravity_x" in intercepted_vars
                and "pde_gravity_y" in intercepted_vars
        ):
            gravity_x = tensor_to_numpy(
                intercepted_vars["pde_gravity_x"]
            )
            gravity_y = tensor_to_numpy(
                intercepted_vars["pde_gravity_y"]
            )
            attention_maps["pde_gravity_magnitude"] = np.sqrt(
                gravity_x ** 2 + gravity_y ** 2 + 1e-8
            )

        return output

    decode_head.forward = types.MethodType(
        intercepted_forward,
        decode_head
    )

    return hook_handles

def extract_first_map(local_array):
    if local_array is None:
        return None

    array = np.asarray(local_array)

    if array.ndim == 4:
        # [B, C, H, W]
        return array[0, 0].astype(np.float32)

    if array.ndim == 3:
        # [B, H, W] or [C, H, W]
        return array[0].astype(np.float32)

    if array.ndim == 2:
        return array.astype(np.float32)

    raise ValueError(
        f"无法解析中间特征图形状: {array.shape}"
    )

def resize_local_map(local_map, target_size, is_binary):
    target_height, target_width = target_size

    if local_map.shape == (target_height, target_width):
        if is_binary:
            return (local_map > 0.5).astype(np.float32)

        return local_map.astype(np.float32)

    tensor = torch.from_numpy(
        local_map.astype(np.float32)
    ).unsqueeze(0).unsqueeze(0)

    if is_binary:
        resized = F.interpolate(
            tensor,
            size=(target_height, target_width),
            mode="nearest"
        )
    else:
        resized = F.interpolate(
            tensor,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False
        )

    resized = resized.squeeze(0).squeeze(0).numpy()

    if is_binary:
        resized = (resized > 0.5).astype(np.float32)

    return resized.astype(np.float32)

def assemble_continuous_map(
    global_sum,
    local_array,
    y,
    x,
    crop_size,
    weight_mask
):
    if local_array is None:
        return

    local_map = extract_first_map(local_array)
    local_map = resize_local_map(
        local_map,
        target_size=(crop_size, crop_size),
        is_binary=False
    )

    global_sum[
        y:y + crop_size,
        x:x + crop_size
    ] += local_map * weight_mask

def assemble_binary_map(
    global_binary,
    local_array,
    y,
    x,
    crop_size
):
    if local_array is None:
        return

    local_map = extract_first_map(local_array)
    local_map = resize_local_map(
        local_map,
        target_size=(crop_size, crop_size),
        is_binary=True
    )

    current_region = global_binary[
        y:y + crop_size,
        x:x + crop_size
    ]

    global_binary[
        y:y + crop_size,
        x:x + crop_size
    ] = np.maximum(current_region, local_map)

def get_output_stem():
    return os.path.splitext(
        os.path.basename(INPUT_IMG_PATH)
    )[0]

def normalize_continuous_map(
    target_map,
    lower_percentile=1,
    upper_percentile=99,
    fixed_range=None
):
    finite_mask = np.isfinite(target_map)

    if not finite_mask.any():
        return np.zeros_like(target_map, dtype=np.uint8)

    clean_map = np.where(finite_mask, target_map, 0.0)

    if fixed_range is not None:
        value_min, value_max = fixed_range
    else:
        valid_values = clean_map[finite_mask]
        value_min = float(
            np.percentile(valid_values, lower_percentile)
        )
        value_max = float(
            np.percentile(valid_values, upper_percentile)
        )

    if value_max <= value_min:
        return np.zeros_like(clean_map, dtype=np.uint8)

    normalized = np.clip(clean_map, value_min, value_max)
    normalized = (
        (normalized - value_min)
        / (value_max - value_min)
        * 255.0
    )

    return normalized.astype(np.uint8)

def save_continuous_heatmap(
    map_sum,
    count_map,
    name_suffix,
    fixed_range=None,
    colormap=cv2.COLORMAP_JET
):
    target_map = map_sum / np.maximum(count_map, 1e-5)

    norm_map = normalize_continuous_map(
        target_map,
        fixed_range=fixed_range
    )

    heatmap = cv2.applyColorMap(norm_map, colormap)

    save_path = os.path.join(
        MAPS_DIR,
        f"{get_output_stem()}_{name_suffix}.png"
    )

    cv2.imwrite(save_path, heatmap)
    print(f"连续响应图已保存至: {save_path}")

    npy_path = os.path.join(
        MAPS_DIR,
        f"{get_output_stem()}_{name_suffix}.npy"
    )
    np.save(npy_path, target_map.astype(np.float32))

def save_gravity_heatmap(map_sum, count_map, name_suffix):
    target_map = map_sum / np.maximum(count_map, 1e-5)

    norm_map = normalize_continuous_map(
        target_map,
        lower_percentile=2,
        upper_percentile=98
    )

    heatmap = cv2.applyColorMap(
        norm_map,
        cv2.COLORMAP_JET
    )

    save_path = os.path.join(
        MAPS_DIR,
        f"{get_output_stem()}_{name_suffix}.png"
    )

    cv2.imwrite(save_path, heatmap)
    print(f"物理场热力图已保存至: {save_path}")

    np.save(
        os.path.join(
            MAPS_DIR,
            f"{get_output_stem()}_{name_suffix}.npy"
        ),
        target_map.astype(np.float32)
    )

def save_sdf_map(map_sum, count_map, name_suffix):
    target_map = map_sum / np.maximum(count_map, 1e-5)

    safe_target = np.clip(target_map, -10.0, 10.0)
    thickness = 0.5
    boundary_map = np.exp(
        -(safe_target ** 2) / thickness
    )

    norm_map = np.uint8(
        np.clip(boundary_map, 0.0, 1.0) * 255
    )

    heatmap = cv2.applyColorMap(
        norm_map,
        cv2.COLORMAP_JET
    )

    save_path = os.path.join(
        MAPS_DIR,
        f"{get_output_stem()}_{name_suffix}.png"
    )

    cv2.imwrite(save_path, heatmap)
    print(f"SDF示意图已保存至: {save_path}")

    np.save(
        os.path.join(
            MAPS_DIR,
            f"{get_output_stem()}_{name_suffix}.npy"
        ),
        target_map.astype(np.float32)
    )

def save_grayscale_map(
    map_sum,
    count_map,
    name_suffix,
    fixed_range=None
):
    target_map = map_sum / np.maximum(count_map, 1e-5)

    norm_map = normalize_continuous_map(
        target_map,
        fixed_range=fixed_range
    )

    save_path = os.path.join(
        MAPS_DIR,
        f"{get_output_stem()}_{name_suffix}.png"
    )

    cv2.imwrite(save_path, norm_map)
    print(f"灰度图已保存至: {save_path}")

    np.save(
        os.path.join(
            MAPS_DIR,
            f"{get_output_stem()}_{name_suffix}.npy"
        ),
        target_map.astype(np.float32)
    )

def save_binary_mask(
    binary_map,
    name_suffix,
    save_color_preview=True
):
    binary_map = (
        binary_map > 0.5
    ).astype(np.uint8)

    grayscale = binary_map * 255

    gray_path = os.path.join(
        MAPS_DIR,
        f"{get_output_stem()}_{name_suffix}.png"
    )

    cv2.imwrite(gray_path, grayscale)
    print(f"二值掩膜已保存至: {gray_path}")

    if save_color_preview:
        color_map = cv2.applyColorMap(
            grayscale,
            cv2.COLORMAP_JET
        )

        color_path = os.path.join(
            MAPS_DIR,
            f"{get_output_stem()}_{name_suffix}_Color.png"
        )

        cv2.imwrite(color_path, color_map)
        print(f"二值掩膜彩色预览已保存至: {color_path}")

    np.save(
        os.path.join(
            MAPS_DIR,
            f"{get_output_stem()}_{name_suffix}.npy"
        ),
        binary_map.astype(np.uint8)
    )

def compute_boundary_iou_tif(
    pred,
    gt,
    boundary_width=5,
    ignore_index=255
):
    def get_inner_boundary(mask, width):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (width, width)
        )

        eroded = cv2.erode(
            mask.astype(np.uint8),
            kernel
        )

        return (mask > 0) & (eroded == 0)

    valid_mask = gt != ignore_index

    pred_boundary = (
        get_inner_boundary(pred == 1, boundary_width)
        & valid_mask
    )

    gt_boundary = (
        get_inner_boundary(gt == 1, boundary_width)
        & valid_mask
    )

    intersection = (
        pred_boundary & gt_boundary
    ).sum()

    union = (
        pred_boundary | gt_boundary
    ).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)

def inference_large_image():
    model = init_model(
        CONFIG_FILE,
        CHECKPOINT_FILE,
        device=DEVICE
    )
    model.eval()

    print("=" * 70)
    print("Decode head class:")
    print(type(model.decode_head))
    print("Decode head source:")
    print(inspect.getfile(type(model.decode_head)))
    print("=" * 70)

    expected_filename = "ufda_dgem_srgformer_head_PDES.py"
    actual_source = inspect.getfile(type(model.decode_head))

    if expected_filename.lower() not in actual_source.lower():
        print(
            "警告：当前加载的解码头似乎不是修改后的PDES源码。"
        )
        print(f"期望文件名: {expected_filename}")
        print(f"实际文件路径: {actual_source}")

    attention_maps = create_attention_storage()
    hook_handles = register_interception_hooks(
        model,
        attention_maps
    )

    image, geo_transform, projection = read_geotiff(
        INPUT_IMG_PATH
    )

    image = preprocess_image(image)
    image, original_height, original_width = (
        pad_image_if_needed(image, CROP_SIZE)
    )

    _, height, width = image.shape

    probability_sum = np.zeros(
        (NUM_CLASSES, height, width),
        dtype=np.float32
    )

    count_map = np.zeros(
        (height, width),
        dtype=np.float32
    )

    #Continuous response maps
    continuous_globals = {
        "nfim1": np.zeros((height, width), dtype=np.float32),
        "nfim2": np.zeros((height, width), dtype=np.float32),
        "ufda_entropy": np.zeros(
            (height, width), dtype=np.float32
        ),
        "ufda_spatial": np.zeros(
            (height, width), dtype=np.float32
        ),
        "ufda_physical": np.zeros(
            (height, width), dtype=np.float32
        ),
        "ufda_joint_base": np.zeros(
            (height, width), dtype=np.float32
        ),
        "joint_mask_highres": np.zeros(
            (height, width), dtype=np.float32
        ),
        "water_probability": np.zeros(
            (height, width), dtype=np.float32
        ),
        "dgem_gravity": np.zeros(
            (height, width), dtype=np.float32
        ),
        "checking_nir_base": np.zeros(
            (height, width), dtype=np.float32
        ),
        "checking_ndwi": np.zeros(
            (height, width), dtype=np.float32
        ),
        "corrected_gravity_field": np.zeros(
            (height, width), dtype=np.float32
        ),
        "pde_phi_initial": np.zeros(
            (height, width), dtype=np.float32
        ),
        "pde_phi_final": np.zeros(
            (height, width), dtype=np.float32
        ),
        "pde_phi_delta": np.zeros(
            (height, width), dtype=np.float32
        ),
        "pde_diffusion_coeff": np.zeros(
            (height, width), dtype=np.float32
        ),
        "pde_gravity_x": np.zeros(
            (height, width), dtype=np.float32
        ),
        "pde_gravity_y": np.zeros(
            (height, width), dtype=np.float32
        ),
        "pde_gravity_magnitude": np.zeros(
            (height, width), dtype=np.float32
        ),
    }

    #Binary masks
    binary_globals = {
        "semantic_water_mask": np.zeros(
            (height, width), dtype=np.float32
        ),
        "semantic_background_mask": np.zeros(
            (height, width), dtype=np.float32
        ),
        "physical_water_mask": np.zeros(
            (height, width), dtype=np.float32
        ),
        "physical_background_mask": np.zeros(
            (height, width), dtype=np.float32
        ),
        "high_uncertainty_mask": np.zeros(
            (height, width), dtype=np.float32
        ),
        "explicit_conflict_mask": np.zeros(
            (height, width), dtype=np.float32
        ),
        "potential_conflict_mask": np.zeros(
            (height, width), dtype=np.float32
        ),
        "initial_conflict_mask": np.zeros(
            (height, width), dtype=np.float32
        ),
        "reliable_conflict_mask": np.zeros(
            (height, width), dtype=np.float32
        ),
        "restricted_evolution_zone": np.zeros(
            (height, width), dtype=np.float32
        ),
    }

    weight_mask = get_weight_mask(CROP_SIZE)

    y_positions = get_sliding_positions(
        height,
        CROP_SIZE,
        STRIDE
    )

    x_positions = get_sliding_positions(
        width,
        CROP_SIZE,
        STRIDE
    )

    print(
        f"滑窗推理中，图像尺寸: {width}x{height}，"
        f"窗口数量: {len(y_positions) * len(x_positions)}"
    )

    with torch.no_grad():
        for y in tqdm(y_positions):
            for x in x_positions:
                # Clear values from the preceding tile. This prevents stale
                # maps from being reused when an optional map is not emitted.
                for key in attention_maps:
                    attention_maps[key] = None

                image_crop = image[
                    :,
                    y:y + CROP_SIZE,
                    x:x + CROP_SIZE
                ]

                input_tensor = torch.from_numpy(
                    image_crop
                ).unsqueeze(0).float().to(DEVICE)

                data_sample = SegDataSample()
                data_sample.set_metainfo(
                    {
                        "img_shape": (CROP_SIZE, CROP_SIZE),
                        "ori_shape": (CROP_SIZE, CROP_SIZE),
                        "pad_shape": (CROP_SIZE, CROP_SIZE),
                    }
                )

                data_input = {
                    "inputs": input_tensor,
                    "data_samples": [data_sample],
                }

                processed_data = model.data_preprocessor(
                    data_input,
                    training=False
                )

                predictions = model.predict(
                    processed_data["inputs"],
                    processed_data["data_samples"]
                )

                logits = (
                    predictions[0]
                    .seg_logits.data
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

                if logits.shape[-2:] != (
                    CROP_SIZE,
                    CROP_SIZE
                ):
                    logits_tensor = torch.from_numpy(
                        logits
                    ).unsqueeze(0)

                    logits = F.interpolate(
                        logits_tensor,
                        size=(CROP_SIZE, CROP_SIZE),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0).numpy()

                probability_sum[
                    :,
                    y:y + CROP_SIZE,
                    x:x + CROP_SIZE
                ] += logits * weight_mask[None, :, :]

                count_map[
                    y:y + CROP_SIZE,
                    x:x + CROP_SIZE
                ] += weight_mask

                for key, global_map in continuous_globals.items():
                    assemble_continuous_map(
                        global_sum=global_map,
                        local_array=attention_maps.get(key),
                        y=y,
                        x=x,
                        crop_size=CROP_SIZE,
                        weight_mask=weight_mask
                    )

                for key, global_map in binary_globals.items():
                    assemble_binary_map(
                        global_binary=global_map,
                        local_array=attention_maps.get(key),
                        y=y,
                        x=x,
                        crop_size=CROP_SIZE
                    )

    #Remove padding if the input image was smaller than one crop.
    probability_sum = probability_sum[
        :,
        :original_height,
        :original_width
    ]

    count_map = count_map[
        :original_height,
        :original_width
    ]

    for key in continuous_globals:
        continuous_globals[key] = continuous_globals[key][
            :original_height,
            :original_width
        ]

    for key in binary_globals:
        binary_globals[key] = binary_globals[key][
            :original_height,
            :original_width
        ]

    #The common denominator does not affect argmax, but explicit
    #normalization keeps the stitched scores well-defined.
    normalized_scores = (
        probability_sum
        / np.maximum(count_map[None, :, :], 1e-5)
    )

    pred_mask = np.argmax(
        normalized_scores,
        axis=0
    ).astype(np.uint8)

    save_geotiff(
        pred_mask,
        OUTPUT_PATH,
        geo_transform,
        projection
    )

    output_png_path = os.path.splitext(
        OUTPUT_PATH
    )[0] + ".png"

    cv2.imwrite(
        output_png_path,
        pred_mask * 255
    )

    print(f"最终分割结果PNG已保存至: {output_png_path}")

    if GT_IMG_PATH and os.path.exists(GT_IMG_PATH):
        print("正在读取真值标签并计算指标")

        gt_mask = cv2.imread(
            GT_IMG_PATH,
            cv2.IMREAD_GRAYSCALE
        )

        if gt_mask is None:
            print(
                f"无法读取真值标签{GT_IMG_PATH}，跳过评测"
            )
        elif gt_mask.shape != (
            original_height,
            original_width
        ):
            print(
                f"GT尺寸{gt_mask.shape}与预测尺寸"
                f"{(original_height, original_width)}不一致，"
                "跳过评测"
            )
        else:
            valid_area = gt_mask != 255

            intersection = np.logical_and(
                np.logical_and(
                    pred_mask == 1,
                    gt_mask == 1
                ),
                valid_area
            ).sum()

            union = np.logical_and(
                np.logical_or(
                    pred_mask == 1,
                    gt_mask == 1
                ),
                valid_area
            ).sum()

            global_iou = intersection / (union + 1e-6)

            boundary_iou = compute_boundary_iou_tif(
                pred_mask,
                gt_mask,
                boundary_width=5,
                ignore_index=255
            )

            print("-" * 60)
            print(
                f"全局Water IoU: {global_iou:.4f} "
                f"({global_iou * 100:.2f}%)"
            )
            print(
                f"Boundary IoU: {boundary_iou:.4f} "
                f"({boundary_iou * 100:.2f}%)"
            )
            print("-" * 60)
    else:
        print("未提供有效GT路径，跳过IoU评测")

    continuous_visualization_config = {
        "nfim1": ("NFIM_1", None),
        "nfim2": ("NFIM_2", None),
        "ufda_entropy": ("TPUD_Entropy", (0.0, 1.0)),
        "ufda_spatial": ("TPUD_Spatial", (0.0, 1.0)),
        "ufda_physical": ("TPUD_Physical", (0.0, 1.0)),
        "ufda_joint_base": (
            "TPUD_Joint_Uncertainty_Base",
            (0.0, 1.0)
        ),
        "joint_mask_highres": (
            "TPUD_Joint_Uncertainty_HighRes",
            (0.0, 1.0)
        ),
        "water_probability": (
            "Auxiliary_Water_Probability",
            (0.0, 1.0)
        ),
        "checking_ndwi": (
            "DGEM_Checking_NDWI_Index",
            None
        ),
    }

    for key, (
        output_name,
        fixed_range
    ) in continuous_visualization_config.items():
        map_data = continuous_globals[key]

        if np.max(np.abs(map_data)) > 0:
            save_continuous_heatmap(
                map_sum=map_data,
                count_map=count_map,
                name_suffix=output_name,
                fixed_range=fixed_range
            )

    if np.max(
        np.abs(continuous_globals["dgem_gravity"])
    ) > 0:
        save_gravity_heatmap(
            map_sum=continuous_globals["dgem_gravity"],
            count_map=count_map,
            name_suffix="DGEM_Gravity_Field"
        )

    for key, output_name in {
        "pde_phi_initial": "PDE_Initial_Log_Odds",
        "pde_phi_final": "PDE_Final_Log_Odds",
        "pde_phi_delta": "PDE_Log_Odds_Increment",
    }.items():
        if np.max(np.abs(continuous_globals[key])) > 0:
            save_continuous_heatmap(
                map_sum=continuous_globals[key],
                count_map=count_map,
                name_suffix=output_name,
                fixed_range=None,
                colormap=cv2.COLORMAP_TURBO
            )

    if np.max(np.abs(
            continuous_globals["corrected_gravity_field"]
    )) > 0:
        save_continuous_heatmap(
            map_sum=continuous_globals["corrected_gravity_field"],
            count_map=count_map,
            name_suffix="DGEM_Corrected_Gravity_Field_Fsharp",
            fixed_range=(-2.5, 1.0),
            colormap=cv2.COLORMAP_TURBO
        )

    if np.max(np.abs(
            continuous_globals["pde_diffusion_coeff"]
    )) > 0:
        save_continuous_heatmap(
            map_sum=continuous_globals["pde_diffusion_coeff"],
            count_map=count_map,
            name_suffix="PDE_Diffusion_Coefficient",
            fixed_range=(0.0, 1.0),
            colormap=cv2.COLORMAP_TURBO
        )

    for key, output_name in {
        "pde_gravity_x": "PDE_Gravity_X",
        "pde_gravity_y": "PDE_Gravity_Y",
        "pde_gravity_magnitude": "PDE_Gravity_Magnitude",
    }.items():
        if np.max(np.abs(continuous_globals[key])) > 0:
            save_continuous_heatmap(
                map_sum=continuous_globals[key],
                count_map=count_map,
                name_suffix=output_name,
                fixed_range=None,
                colormap=cv2.COLORMAP_TURBO
            )

    if np.max(
        np.abs(continuous_globals["checking_nir_base"])
    ) > 0:
        save_grayscale_map(
            map_sum=continuous_globals[
                "checking_nir_base"
            ],
            count_map=count_map,
            name_suffix="DGEM_Checking_NIR_Base",
            fixed_range=(0.0, 1.0)
        )

    binary_visualization_config = {
        "semantic_water_mask":
            "Semantic_Water_Mask",
        "semantic_background_mask":
            "Semantic_Background_Mask",
        "physical_water_mask":
            "Physical_Water_Support_Mask",
        "physical_background_mask":
            "Physical_Background_Support_Mask",
        "high_uncertainty_mask":
            "TPUD_High_Uncertainty_Mask",
        "explicit_conflict_mask":
            "DGEM_Explicit_Conflict_Mask",
        "potential_conflict_mask":
            "DGEM_Potential_Conflict_Mask",
        "initial_conflict_mask":
            "DGEM_Initial_Conflict_Mask",
        "reliable_conflict_mask":
            "DGEM_Reliable_Conflict_Mask",
        "restricted_evolution_zone":
            "DGEM_Restricted_Evolution_Zone",
    }

    for key, output_name in (
        binary_visualization_config.items()
    ):
        map_data = binary_globals[key]

        # M_phy may be empty only if F_sharp is exactly zero everywhere.
        # Other empty masks are still meaningful, so save all binary maps.
        save_binary_mask(
            binary_map=map_data,
            name_suffix=output_name,
            save_color_preview=True
        )

    # Remove registered native hooks.
    for handle in hook_handles:
        handle.remove()

    print("=" * 70)
    print("推理和可视化完成")
    print(f"结果目录: {MAPS_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    inference_large_image()
