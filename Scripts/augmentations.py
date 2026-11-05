"""
Offline augmentation for YOLO segmentation (polygon) datasets: MixUp, CutMix,
Mosaic and Albumentations transforms.
"""

import os
import cv2
import numpy as np
import random
import albumentations as A
import logging
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

#####################
# Configuration Parameters
#####################

SRC_ROOT = "/path/to/MyDataset"        # dataset in the YOLO layout described above
DST_ROOT = "/path/to/MyDataset_Aug"    # output folder (created if it does not exist)

CLASS_NAMES = ["mycoral"]              # class name, as in the dataset's data.yaml

SPLITS_TO_AUGMENT = ["train"]            # augmented
SPLITS_TO_COPY_ONLY = ["valid", "test"]  # copied unchanged, never augmented

N_FOLD = 15  # Number of augmented copies per training image (the original is saved as well)

# Advanced augmentation selection probabilities.
P_MIXUP = 0.33
P_CUTMIX = 0.33
P_MOSAIC = 0.34

# Smallest polygon kept when an augmented mask is converted back to YOLO
# polygons. This is not an augmentation setting: it only removes tiny fragments
# that can appear after MixUp blending or rotation.
MIN_POLYGON_AREA_PX = 4
MIN_POLYGON_POINTS = 3

#####################
# Logging Setup
#####################
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


#####################
# YOLO-seg polygon <-> instance-mask helpers
#####################

def read_yolo_polygons(label_path):
    """
    Returns list of (class_id, [(x_norm, y_norm), ...]) for every instance
    line in a YOLO-seg .txt label file. Empty list if the file is missing
    or has no valid polygon lines (background-only image).
    """
    polys = []
    if not os.path.exists(label_path):
        return polys
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class id + at least 3 (x, y) pairs
                continue
            cls = int(float(parts[0]))
            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                coords = coords[:-1]
            pts = list(zip(coords[0::2], coords[1::2]))
            if len(pts) >= 3:
                polys.append((cls, pts))
    return polys


def polygons_to_instance_mask(polys, h, w):
    """
    Rasterizes polygons into a uint16 instance mask: 0 = background,
    1..N = one id per polygon instance (paint order = list order).
    """
    mask = np.zeros((h, w), dtype=np.uint16)
    for idx, (cls, pts) in enumerate(polys, start=1):
        abs_pts = np.array(
            [[int(round(x * w)), int(round(y * h))] for x, y in pts],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [abs_pts], color=int(idx))
    return mask


def mask_to_yolo_polygons(mask, class_id=0):
    """
    Extracts polygon(s) per unique nonzero instance id from an (augmented)
    instance mask and returns them as YOLO-seg label lines
    ("<class_id> x1 y1 x2 y2 ...", normalized 0-1).
    An instance that got split (e.g. by cutmix/mosaic clipping or rotation)
    yields one line per resulting contour.
    """
    h, w = mask.shape[:2]
    lines = []
    for inst_id in np.unique(mask):
        if inst_id == 0:
            continue
        inst_bin = (mask == inst_id).astype(np.uint8)
        contours, _ = cv2.findContours(inst_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_POLYGON_AREA_PX or len(cnt) < MIN_POLYGON_POINTS:
                continue
            cnt = cnt.reshape(-1, 2)
            norm = []
            for x, y in cnt:
                norm.append(min(max(x / w, 0.0), 1.0))
                norm.append(min(max(y / h, 0.0), 1.0))
            lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in norm))
    return lines


def write_yolo_label(lines, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


#####################
# Advanced Augmentation Functions
#####################
def mixup(image1, image2, mask1, mask2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    mixed_image = (lam * image1.astype(np.float32) + (1 - lam) * image2.astype(np.float32)).astype(np.uint8)
    rand_matrix = np.random.rand(*mask1.shape)
    mixed_mask = np.where(rand_matrix < lam, mask1, mask2).astype(mask1.dtype)
    return mixed_image, mixed_mask


def cutmix(image1, image2, mask1, mask2, alpha=0.4):
    H, W, _ = image1.shape
    lam = np.random.beta(alpha, alpha)
    cut_ratio = np.sqrt(1 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    cutmix_image = image1.copy()
    cutmix_mask = mask1.copy()
    cutmix_image[y1:y2, x1:x2] = image2[y1:y2, x1:x2]
    cutmix_mask[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]
    return cutmix_image, cutmix_mask


def mosaic_augmentation(img_list, mask_list, boundary_width=10):
    h, w, _ = img_list[0].shape

    # Mosaic partners are picked at random from the same split. If a partner
    # has a different size from the first image, it is resized to match.
    fixed_imgs, fixed_masks = [img_list[0]], [mask_list[0]]
    for im, mk in zip(img_list[1:], mask_list[1:]):
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            mk = cv2.resize(mk, (w, h), interpolation=cv2.INTER_NEAREST)
        fixed_imgs.append(im)
        fixed_masks.append(mk)
    img_list, mask_list = fixed_imgs, fixed_masks

    cx = random.randint(int(w * 0.25), int(w * 0.75))
    cy = random.randint(int(h * 0.25), int(h * 0.75))

    mosaic_img = np.zeros_like(img_list[0])
    mosaic_mask = np.zeros_like(mask_list[0])

    mosaic_img[0:cy, 0:cx] = img_list[0][h - cy:h, w - cx:w]
    mosaic_mask[0:cy, 0:cx] = mask_list[0][h - cy:h, w - cx:w]
    mosaic_img[0:cy, cx:w] = img_list[1][h - cy:h, 0:w - cx]
    mosaic_mask[0:cy, cx:w] = mask_list[1][h - cy:h, 0:w - cx]
    mosaic_img[cy:h, 0:cx] = img_list[2][0:h - cy, w - cx:w]
    mosaic_mask[cy:h, 0:cx] = mask_list[2][0:h - cy, w - cx:w]
    mosaic_img[cy:h, cx:w] = img_list[3][0:h - cy, 0:w - cx]
    mosaic_mask[cy:h, cx:w] = mask_list[3][0:h - cy, 0:w - cx]

    if boundary_width > 0:
        kernel_size = boundary_width if boundary_width % 2 == 1 else boundary_width + 1
        # Vertical feathering.
        x1 = max(cx - boundary_width, 0)
        x2 = min(cx + boundary_width, w)
        boundary_region = mosaic_img[:, x1:x2]
        blurred_region = cv2.GaussianBlur(boundary_region, (kernel_size, kernel_size), 0)
        alpha_mask = np.linspace(0, 1, x2 - x1).reshape(1, -1, 1)
        mosaic_img[:, x1:x2] = (mosaic_img[:, x1:x2].astype(np.float32) * (1 - alpha_mask) +
                                blurred_region.astype(np.float32) * alpha_mask).astype(np.uint8)
        # Horizontal feathering.
        y1 = max(cy - boundary_width, 0)
        y2 = min(cy + boundary_width, h)
        boundary_region = mosaic_img[y1:y2, :]
        blurred_region = cv2.GaussianBlur(boundary_region, (kernel_size, kernel_size), 0)
        alpha_mask = np.linspace(0, 1, y2 - y1).reshape(-1, 1, 1)
        mosaic_img[y1:y2, :] = (mosaic_img[y1:y2, :].astype(np.float32) * (1 - alpha_mask) +
                                blurred_region.astype(np.float32) * alpha_mask).astype(np.uint8)

    mosaic_img = adjust_color(mosaic_img)
    return mosaic_img, mosaic_mask


def adjust_color(image):
    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)
    ])
    augmented = transform(image=image)
    return augmented["image"]


#####################
# Basic Augmentations using Albumentations
#####################
# Albumentations 2.0 replaced GaussNoise's `var_limit` with `std_range` and
# removed ElasticTransform's `alpha_affine`. Passing the old arguments to 2.x
# does not raise an error: the transform falls back to its default range, which
# for GaussNoise is a much stronger noise than var_limit=(10, 50). The helpers
# below detect the installed version and convert the settings, so the noise
# and distortion strength is the same in both versions.
_ALB_MAJOR = int(A.__version__.split(".")[0])


def _make_gauss_noise():
    if _ALB_MAJOR >= 2:
        # old: var_limit=(10.0, 50.0) -> std = sqrt(var); new API's
        # std_range is normalized as a fraction of the 8-bit max (255).
        std_lo = (10.0 ** 0.5) / 255.0
        std_hi = (50.0 ** 0.5) / 255.0
        return A.GaussNoise(std_range=(std_lo, std_hi), p=0.5)
    return A.GaussNoise(var_limit=(10.0, 50.0), p=0.5)


def _make_elastic_transform():
    if _ALB_MAJOR >= 2:
        # alpha_affine has no equivalent in >=2.0 (the extra affine jitter
        # component was removed from ElasticTransform entirely); alpha and
        # sigma still mean the same thing, so those are preserved exactly.
        return A.ElasticTransform(alpha=34, sigma=4, p=0.5)
    return A.ElasticTransform(alpha=34, sigma=4, alpha_affine=4, p=0.5)


logging.info(
    f"albumentations version {A.__version__} detected -> "
    f"{'using converted std_range / dropping alpha_affine (no >=2.0 equivalent)' if _ALB_MAJOR >= 2 else 'using original var_limit / alpha_affine'}"
)

# Built once and reused for every image.
_BASIC_TRANSFORM = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30,
                       p=0.7, border_mode=cv2.BORDER_REFLECT_101),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.RandomGamma(gamma_limit=(80, 120), p=0.7),
    _make_gauss_noise(),
    A.GaussianBlur(blur_limit=(3, 5), p=0.5),
    _make_elastic_transform(),
], additional_targets={'mask': 'mask'})


def apply_basic_augmentations(image, mask):
    augmented = _BASIC_TRANSFORM(image=image, mask=mask)
    return augmented['image'], augmented['mask']


#####################
# Preloading Functions (the mask is built from the YOLO .txt label)
#####################
def load_image_and_mask(filename, orig_img_dir, orig_label_dir):
    base_name = os.path.splitext(filename)[0]
    image_path = os.path.join(orig_img_dir, filename)
    label_path = os.path.join(orig_label_dir, base_name + ".txt")

    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f"Preload skipped for {filename}: could not read image.")
        return filename, None, None

    h, w = image.shape[:2]
    polys = read_yolo_polygons(label_path)
    mask = polygons_to_instance_mask(polys, h, w)
    return filename, image, mask


#####################
# Worker Functions for Parallel Processing
#####################
def save_original_image(filename, images, masks, aug_img_dir, aug_label_dir):
    base_name = os.path.splitext(filename)[0]
    image = images.get(filename)
    mask = masks.get(filename)
    if image is None or mask is None:
        logging.warning(f"Skipping saving original for {filename}: image or mask not loaded.")
        return
    cv2.imwrite(os.path.join(aug_img_dir, f"{base_name}_orig.jpg"), image)
    lines = mask_to_yolo_polygons(mask, class_id=0)
    write_yolo_label(lines, os.path.join(aug_label_dir, f"{base_name}_orig.txt"))
    logging.info(f"Saved original {filename}.")


def process_image(filename, images, masks, all_filenames, aug_img_dir, aug_label_dir):
    base_name = os.path.splitext(filename)[0]
    image = images.get(filename)
    mask = masks.get(filename)
    if image is None or mask is None:
        logging.warning(f"Skipping augmentation for {filename}: image or mask not loaded.")
        return

    # Exclude self for partner selection.
    others = [f for f in all_filenames if f != filename]
    for i in range(N_FOLD):
        aug_type = "basic"
        aug_image = image.copy()
        aug_mask = mask.copy()

        candidate_methods = []
        if random.random() < P_MIXUP and others:
            candidate_methods.append("mixup")
        if random.random() < P_CUTMIX and others:
            candidate_methods.append("cutmix")
        if random.random() < P_MOSAIC and len(all_filenames) >= 4:
            candidate_methods.append("mosaic")

        if candidate_methods:
            chosen = random.choice(candidate_methods)
            if chosen == "mixup":
                partner = random.choice(others)
                partner_image = images.get(partner)
                partner_mask = masks.get(partner)
                if partner_image is not None and partner_mask is not None:
                    aug_image, aug_mask = mixup(image, partner_image, mask, partner_mask)
                    aug_type = "mixup"
            elif chosen == "cutmix":
                partner = random.choice(others)
                partner_image = images.get(partner)
                partner_mask = masks.get(partner)
                if partner_image is not None and partner_mask is not None:
                    aug_image, aug_mask = cutmix(image, partner_image, mask, partner_mask)
                    aug_type = "cutmix"
            elif chosen == "mosaic":
                if len(others) >= 3:
                    mosaic_files = random.sample(others, 3)
                    mosaic_images = [image] + [images.get(f) for f in mosaic_files if images.get(f) is not None]
                    mosaic_masks = [mask] + [masks.get(f) for f in mosaic_files if masks.get(f) is not None]
                    if len(mosaic_images) == 4 and len(mosaic_masks) == 4:
                        aug_image, aug_mask = mosaic_augmentation(mosaic_images, mosaic_masks, boundary_width=10)
                        aug_type = "mosaic"
        # Always apply basic augmentations.
        aug_image, aug_mask = apply_basic_augmentations(aug_image, aug_mask)
        out_img_path = os.path.join(aug_img_dir, f"{base_name}_aug_{aug_type}_{i}.jpg")
        out_label_path = os.path.join(aug_label_dir, f"{base_name}_aug_{aug_type}_{i}.txt")
        cv2.imwrite(out_img_path, aug_image)
        lines = mask_to_yolo_polygons(aug_mask, class_id=0)
        write_yolo_label(lines, out_label_path)
        logging.info(f"Processed {filename} with {aug_type} augmentation ({i + 1}/{N_FOLD}).")


#####################
# Process one split: augment (train) or copy-through (valid/test)
#####################
def process_augmented_split(split_name, src_root, dst_root):
    logging.info(f"Augmenting '{split_name}' split...")
    orig_img_dir = os.path.join(src_root, split_name, "images")
    orig_label_dir = os.path.join(src_root, split_name, "labels")

    aug_img_dir = os.path.join(dst_root, split_name, "images")
    aug_label_dir = os.path.join(dst_root, split_name, "labels")
    os.makedirs(aug_img_dir, exist_ok=True)
    os.makedirs(aug_label_dir, exist_ok=True)

    img_files = [f for f in os.listdir(orig_img_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    logging.info(f"Found {len(img_files)} images in '{split_name}'.")

    # Preload images and masks concurrently.
    images, masks = {}, {}
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_image_and_mask, f, orig_img_dir, orig_label_dir) for f in img_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Preloading {split_name}"):
            fname, img, msk = future.result()
            if img is not None and msk is not None:
                images[fname] = img
                masks[fname] = msk

    preloaded_files = list(images.keys())
    if not preloaded_files:
        logging.error(f"No images preloaded for '{split_name}'. Skipping this split.")
        return

    # Save originals concurrently.
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(save_original_image, f, images, masks, aug_img_dir, aug_label_dir)
                   for f in preloaded_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Saving originals {split_name}"):
            future.result()

    # Process image augmentations concurrently.
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, f, images, masks, preloaded_files, aug_img_dir, aug_label_dir)
                   for f in preloaded_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Augmenting {split_name}"):
            future.result()


def copy_split_unchanged(split_name, src_root, dst_root):
    """
    Copy a split (validation or test) to the output folder unchanged, so the
    output is a complete dataset ready for training.
    """
    logging.info(f"Copying '{split_name}' split through unchanged (not augmented)...")
    src_img_dir = os.path.join(src_root, split_name, "images")
    src_label_dir = os.path.join(src_root, split_name, "labels")
    dst_img_dir = os.path.join(dst_root, split_name, "images")
    dst_label_dir = os.path.join(dst_root, split_name, "labels")
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)

    img_files = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    for f in tqdm(img_files, desc=f"Copying {split_name}"):
        base = os.path.splitext(f)[0]
        shutil.copy2(os.path.join(src_img_dir, f), os.path.join(dst_img_dir, f))
        label_src = os.path.join(src_label_dir, base + ".txt")
        if os.path.exists(label_src):
            shutil.copy2(label_src, os.path.join(dst_label_dir, base + ".txt"))
    logging.info(f"Copied {len(img_files)} images for '{split_name}'.")


def write_data_yaml(dst_root, class_names):
    content = (
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names!r}\n"
    )
    with open(os.path.join(dst_root, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(content)


#####################
# Main
#####################
def main():
    os.makedirs(DST_ROOT, exist_ok=True)

    for split_name in SPLITS_TO_AUGMENT:
        process_augmented_split(split_name, SRC_ROOT, DST_ROOT)

    for split_name in SPLITS_TO_COPY_ONLY:
        copy_split_unchanged(split_name, SRC_ROOT, DST_ROOT)

    write_data_yaml(DST_ROOT, CLASS_NAMES)
    logging.info(f"DONE. Augmented dataset written to: {DST_ROOT}")


if __name__ == "__main__":
    main()