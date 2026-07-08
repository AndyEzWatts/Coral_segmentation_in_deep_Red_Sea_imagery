import os
import cv2
import numpy as np
import random
import albumentations as A
import logging
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

#####################
# Configuration Parameters
#####################
N_FOLD = 15  # Number of augmentation iterations per image (in addition to saving the original)

# Advanced augmentation selection probabilities.
P_MIXUP = 0.33
P_CUTMIX = 0.33
P_MOSAIC = 0.34

#####################
# Logging Setup
#####################
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


#####################
# Custom Palette Generation
#####################
def generate_palette_from_colormap(n, colormap='Set1'):
    palette = [0, 0, 0]
    list_colors = [
        (228, 26, 28),
        (55, 126, 184),
        (77, 175, 74),
        (152, 78, 163),
        (255, 127, 0),
        (255, 255, 51),
        (166, 86, 40),
        (247, 129, 191),
        (153, 153, 153)
    ]
    for i in range(0, n - 1):
        palette.extend(list_colors[i % len(list_colors)])
    return palette


#####################
# Helper Functions for Reading and Saving Masks
#####################
def load_mask(path):
    try:
        pil_mask = Image.open(path)
        return np.array(pil_mask)
    except Exception as e:
        logging.error(f"Failed to load mask {path}: {e}")
        return None


def save_mask(mask_array, save_path):
    try:
        pil_mask = Image.fromarray(mask_array.astype(np.uint8), mode="P")
        seg_classes = len(np.unique(mask_array))
        palette = generate_palette_from_colormap(seg_classes, colormap='Set1')
        pil_mask.putpalette(palette)
        pil_mask.save(save_path)
    except Exception as e:
        logging.error(f"Failed to save mask {save_path}: {e}")


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
def apply_basic_augmentations(image, mask):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30,
                           p=0.7, border_mode=cv2.BORDER_REFLECT_101),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.RandomGamma(gamma_limit=(80, 120), p=0.7),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        A.ElasticTransform(alpha=34, sigma=4, alpha_affine=4, p=0.5)
    ], additional_targets={'mask': 'mask'})
    augmented = transform(image=image, mask=mask)
    return augmented['image'], augmented['mask']


#####################
# Preloading Functions
#####################
def load_image_and_mask(filename, orig_img_dir, orig_mask_dir):
    base_name = os.path.splitext(filename)[0]
    image_path = os.path.join(orig_img_dir, filename)
    mask_path = os.path.join(orig_mask_dir, base_name + "_mask.png")
    image = cv2.imread(image_path)
    mask = load_mask(mask_path)
    if image is None or mask is None:
        logging.warning(f"Preload skipped for {filename}: image or mask not loaded.")
    return filename, image, mask


#####################
# Worker Functions for Parallel Processing
#####################
def save_original_image(filename, images, masks, aug_img_dir, aug_mask_dir):
    base_name = os.path.splitext(filename)[0]
    image = images.get(filename)
    mask = masks.get(filename)
    if image is None or mask is None:
        logging.warning(f"Skipping saving original for {filename}: image or mask not loaded.")
        return
    cv2.imwrite(os.path.join(aug_img_dir, f"{base_name}_orig.jpg"), image)
    save_mask(mask, os.path.join(aug_mask_dir, f"{base_name}_orig.png"))
    logging.info(f"Saved original {filename}.")


def process_image(filename, images, masks, all_filenames, aug_img_dir, aug_mask_dir):
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
        out_mask_path = os.path.join(aug_mask_dir, f"{base_name}_aug_{aug_type}_{i}.png")
        cv2.imwrite(out_img_path, aug_image)
        save_mask(aug_mask, out_mask_path)
        logging.info(f"Processed {filename} with {aug_type} augmentation ({i + 1}/{N_FOLD}).")


#####################
# New Helper Function: Process one data split
#####################
def process_data_split(dataset_name, data_set, base_dir):
    logging.info(f"Processing {data_set} set of {dataset_name}...")
    orig_dir = os.path.join(base_dir, dataset_name)
    aug_dir = os.path.join(base_dir, f"{dataset_name}_Aug")

    aug_img_dir = os.path.join(aug_dir, "img_dir", data_set)
    aug_mask_dir = os.path.join(aug_dir, "ann_dir", data_set)
    os.makedirs(aug_img_dir, exist_ok=True)
    os.makedirs(aug_mask_dir, exist_ok=True)

    orig_img_dir = os.path.join(orig_dir, "img_dir", data_set)
    orig_mask_dir = os.path.join(orig_dir, "ann_dir", data_set)

    img_files = [f for f in os.listdir(orig_img_dir) if f.lower().endswith('.jpg')]
    logging.info(f"Found {len(img_files)} images in {data_set} set of {dataset_name}.")

    # Preload images and masks concurrently.
    images, masks = {}, {}
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_image_and_mask, f, orig_img_dir, orig_mask_dir) for f in img_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Preloading {data_set} {dataset_name}"):
            fname, img, msk = future.result()
            if img is not None and msk is not None:
                images[fname] = img
                masks[fname] = msk

    preloaded_files = list(images.keys())
    if not preloaded_files:
        logging.error(f"No images preloaded for {dataset_name} {data_set}. Skipping this split.")
        return

    # Save originals concurrently.
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(save_original_image, f, images, masks, aug_img_dir, aug_mask_dir)
                   for f in preloaded_files]
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"Saving originals {data_set} {dataset_name}"):
            future.result()

    # Process image augmentations concurrently.
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, f, images, masks, preloaded_files, aug_img_dir, aug_mask_dir)
                   for f in preloaded_files]
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"Augmenting images {data_set} {dataset_name}"):
            future.result()


#####################
# Main Augmentation Pipeline (Datasets processed concurrently)
#####################
def main():
    global N_FOLD
    import argparse
    parser = argparse.ArgumentParser(
        description="Augment a YOLO segmentation dataset (Mixup, CutMix, Mosaic + Albumentations).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir", required=True,
        help="Directory that contains your dataset folders (e.g. /data/datasets)",
    )
    parser.add_argument(
        "--datasets", nargs="+", required=True,
        help="One or more dataset folder names inside dataset_dir (e.g. Whipcoral Antipatharia)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"],
        help="Data splits to augment",
    )
    parser.add_argument(
        "--n_fold", type=int, default=N_FOLD,
        help="Number of augmentation iterations per image",
    )
    args = parser.parse_args()

    N_FOLD = args.n_fold

    with ThreadPoolExecutor() as executor:
        futures = []
        for dataset_name in args.datasets:
            for data_set in args.splits:
                futures.append(
                    executor.submit(process_data_split, dataset_name, data_set, args.dataset_dir)
                )
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing datasets"):
            future.result()

if __name__ == "__main__":
    main()
