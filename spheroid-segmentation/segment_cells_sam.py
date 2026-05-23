from pathlib import Path

import numpy as np
import imageio.v3 as imageio
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.measure import regionprops, label as skimage_label
from micro_sam import util
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
from micro_sam.prompt_based_segmentation import segment_from_box


# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_DIR     = Path(r"D:\ToxBox\Preprocessing_KIT\Data")
OUTPUT_DIR    = Path(r"D:\ToxBox\Preprocessing_KIT\SAM_Data_segmented")
EMBEDDING_DIR = Path(r"D:\ToxBox\Preprocessing_KIT\SAM_Data_embeddings")

#INPUT_DIR     = Path(r"D:\ToxBox\Preprocessing_KIT\Data_March 2026, HEPG only, Blind test")
#OUTPUT_DIR    = Path(r"D:\ToxBox\Preprocessing_KIT\SAM_Data_segmented_blind")
#EMBEDDING_DIR = Path(r"D:\ToxBox\Preprocessing_KIT\SAM_Data_embeddings_blind")

# ── Normal pipeline settings ───────────────────────────────────────────────────
MAX_ECCENTRICITY = 0.85
MIN_SOLIDITY     = 0.40

# Bounding box extension for normal pipeline SAM pass 2.
NORMAL_BOX_EXTENSION  = 0.20  # ← tune for control/normal folders

# Bounding box extension for PAR200/PAR2000
TEXTURE_BOX_EXTENSION = 0.10 # ← tune for PAR200/PAR2000


# ── Texture folder settings ────────────────────────────────────────────────────
# These folders have textured backgrounds. Instead of AIS auto-segmentation,
# we directly find the darkest large blob (the spheroid) on the ORIGINAL image,
# then use its bounding box as a prompt for SAM.
TEXTURE_FOLDERS = {"PAR200", "PAR2000"}

# Percentile threshold: pixels darker than this are candidates for "spheroid".
# Lower = stricter (only the very darkest pixels). Raise if spheroid is not found.
TEXTURE_DARK_PERCENTILE = 10

# Minimum area for a dark blob to be considered a spheroid candidate.
TEXTURE_MIN_AREA = 30000


# ── Helper functions ───────────────────────────────────────────────────────────

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Invert so spheroids (always darker than background) appear bright for SAM."""
    dtype = image.dtype
    if np.issubdtype(dtype, np.unsignedinteger):
        return np.iinfo(dtype).max - image
    return image.max() - image


def fill_holes(seg: np.ndarray) -> np.ndarray:
    """Fill enclosed holes inside each detected region."""
    filled = np.zeros_like(seg)
    for label_id in range(1, int(seg.max()) + 1):
        filled[binary_fill_holes(seg == label_id)] = label_id
    return filled


def keep_largest_round_region(seg: np.ndarray) -> np.ndarray:
    """Keep only the largest round/compact region. Returns binary 0/1 mask."""
    if seg.max() == 0:
        return seg

    best_label, best_area = None, 0
    for prop in regionprops(seg):
        round_enough   = prop.eccentricity <= MAX_ECCENTRICITY
        compact_enough = prop.solidity     >= MIN_SOLIDITY
        status = "OK" if (round_enough and compact_enough) else "REJECTED"
        print(f"     label {prop.label:3d} | area={prop.area:>8,} | "
              f"eccentricity={prop.eccentricity:.2f} | "
              f"solidity={prop.solidity:.2f} | {status}")
        if round_enough and compact_enough and prop.area > best_area:
            best_area, best_label = prop.area, prop.label

    if best_label is None:
        print("     → no round region passed shape filter")
        return np.zeros_like(seg)

    result = np.zeros_like(seg)
    result[seg == best_label] = 1
    return result


def find_dark_spheroid_box(image_original: np.ndarray):
    """
    Texture-folder strategy: find the darkest, largest, roundest blob on the
    ORIGINAL (non-inverted) image. This is the spheroid.
    Returns bounding box [y_min, x_min, y_max, x_max], or None if not found.
    """
    # Work on grayscale
    gray = image_original if image_original.ndim == 2 else image_original.mean(axis=-1)

    # Threshold: keep only pixels darker than the Nth percentile
    threshold = np.percentile(gray, TEXTURE_DARK_PERCENTILE)
    dark_mask = gray <= threshold

    # Label connected dark regions
    labeled = skimage_label(dark_mask)

    best_prop, best_area = None, 0
    for prop in regionprops(labeled):
        if prop.area < TEXTURE_MIN_AREA:
            continue
        if prop.eccentricity > MAX_ECCENTRICITY:
            continue
        if prop.solidity < MIN_SOLIDITY:
            continue
        if prop.area > best_area:
            best_area = prop.area
            best_prop = prop

    if best_prop is None:
        print(f"     → no dark blob found (threshold={threshold:.1f}, "
              f"try raising TEXTURE_DARK_PERCENTILE or lowering TEXTURE_MIN_AREA)")
        return None

    y_min, x_min, y_max, x_max = best_prop.bbox
    print(f"     dark blob: area={best_prop.area:,}px | "
          f"eccentricity={best_prop.eccentricity:.2f} | "
          f"bbox=[{y_min},{x_min},{y_max},{x_max}]")
    return np.array([y_min, x_min, y_max, x_max])


def get_bounding_box(seg: np.ndarray) -> np.ndarray:
    """Bounding box [y_min, x_min, y_max, x_max] of foreground region."""
    ys, xs = np.where(seg > 0)
    return np.array([ys.min(), xs.min(), ys.max(), xs.max()])


def run_box_sam(predictor, image_embeddings, box, extension: float) -> np.ndarray:
    """Run SAM segmentation using a bounding box prompt. Returns binary mask."""
    mask = segment_from_box(
        predictor,
        box=box,
        image_embeddings=image_embeddings,
        box_extension=extension,
    )
    if mask.ndim == 3:
        mask = mask[0]
    return mask.astype(np.uint8)


def save_results(seg: np.ndarray, output_path: Path):
    """Save binary result TIF (255 = spheroid, 0 = background)."""
    imageio.imwrite(output_path, (seg > 0).astype(np.uint8) * 255)
    print(f"  → Saved: {output_path.name}")


# ── Load model once ────────────────────────────────────────────────────────────
predictor, segmenter = get_predictor_and_segmenter(
    model_type="vit_b",          # model_type="vit_b",  model_type="vit_h", model_type="vit_l",   # plain SAM
    segmentation_mode="amg",        # segmentation_mode="amg" for SAM only
    device="cpu",
)

# ── Process all images ─────────────────────────────────────────────────────────
image_paths = sorted(INPUT_DIR.rglob("*.tif"))
print(f"Found {len(image_paths)} images total.\n")

for img_path in image_paths:
    relative       = img_path.relative_to(INPUT_DIR)
    output_path    = OUTPUT_DIR    / relative
    embedding_path = EMBEDDING_DIR / relative.with_suffix(".zarr")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    is_texture = any(part in TEXTURE_FOLDERS for part in img_path.parts)
    print(f"Processing: {relative}  {'[TEXTURE MODE]' if is_texture else ''}")

    # Always invert for SAM (spheroids darker than background)
    image_original = imageio.imread(img_path)
    image_inverted = preprocess_image(image_original)

    if is_texture:
        # ── TEXTURE PIPELINE ────────────────────────────────────────────
        # Step 1: find the darkest large blob on the ORIGINAL image
        print("  → Step 1: locating spheroid via dark-blob detection")
        box = find_dark_spheroid_box(image_original)

        if box is None:
            print("  → spheroid not found, skipping\n")
            continue

        # Step 2: compute embeddings on INVERTED image (SAM expects bright objects)
        print("  → Step 2: computing embeddings")
        image_embeddings = util.precompute_image_embeddings(
            predictor=predictor,
            input_=image_inverted,
            save_path=embedding_path,
            ndim=2,
            verbose=False,
        )

        # Step 3: SAM box-prompted segmentation
        print(f"  → Step 3: SAM box prompt (extension={TEXTURE_BOX_EXTENSION})")
        seg = run_box_sam(predictor, image_embeddings, box, extension=TEXTURE_BOX_EXTENSION)
        seg = fill_holes(seg)

        area = int(np.sum(seg > 0))
        print(f"  → {area:,} pixels detected")

    else:
        # ── NORMAL PIPELINE ─────────────────────────────────────────────
        # Build generate() kwargs based on segmentation mode:
        # AIS (micro-sam decoder) and AMG (plain SAM grid) use different parameters.
        from micro_sam.instance_segmentation import InstanceSegmentationWithDecoder
        if isinstance(segmenter, InstanceSegmentationWithDecoder):
            generate_kwargs = dict(
                foreground_threshold=0.15,
                boundary_distance_threshold=0.3,
                center_distance_threshold=0.3,
                distance_smoothing=3.0,
                min_size=50000,
            )
        else:  # AMG
            generate_kwargs = dict(
                pred_iou_thresh=0.65,
                stability_score_thresh=0.7,
                min_mask_region_area=50000,
            )

        # Pass 1: automatic segmentation
        seg = automatic_instance_segmentation(
            predictor=predictor,
            segmenter=segmenter,
            input_path=image_inverted,
            output_path=output_path,
            embedding_path=embedding_path,
            ndim=2,
            verbose=True,
            **generate_kwargs,
        )

        if seg is None:
            print("  → no segmentation produced\n")
            continue

        print(f"  → Pass 1: {int(seg.max())} raw candidate(s)")
        seg = fill_holes(seg)
        seg = keep_largest_round_region(seg)

        if seg.max() == 0:
            print("  → nothing passed shape filter\n")
            continue

        # Pass 2: box-prompted SAM refinement
        box = get_bounding_box(seg)
        print(f"  → Pass 2: box prompt [{box[0]},{box[1]},{box[2]},{box[3]}] "
              f"(extension={NORMAL_BOX_EXTENSION})")

        image_embeddings = util.precompute_image_embeddings(
            predictor=predictor,
            input_=image_inverted,
            save_path=embedding_path,
            ndim=2,
            verbose=False,
        )
        refined = fill_holes(run_box_sam(predictor, image_embeddings, box, extension=NORMAL_BOX_EXTENSION))

        original_area = int(np.sum(seg > 0))
        refined_area  = int(np.sum(refined > 0))
        print(f"  → Pass 1: {original_area:,}px | Pass 2: {refined_area:,}px | "
              f"change: {refined_area - original_area:+,}px")

        seg = refined if refined_area >= original_area * 0.8 else seg

    if int(np.sum(seg > 0)) > 0:
        save_results(seg, output_path)
    else:
        print("  → nothing saved")
    print()
