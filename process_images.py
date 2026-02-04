import os
import cv2
import numpy as np


def process_image(img_path, out_path):
    # Read image with OpenCV
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    h, w = img.shape[:2]
    
    # Bottom-right region (scale bar area)
    x1 = int(w * 0.85)  # Right 15% of image
    y1 = int(h * 0.90)  # Bottom 10% of image
    
    # Extract the region
    region = img[y1:h, x1:w].copy()
    
    # Create mask for bright pixels (scale bar and text are usually white/bright)
    if len(region.shape) == 3:  # Color image
        gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:  # Already grayscale
        gray_region = region
    
    # Threshold to find bright pixels (scale bar and text)
    _, mask = cv2.threshold(gray_region, 200, 255, cv2.THRESH_BINARY)
    
    # Dilate mask slightly to cover text/line edges
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Inpaint the region to fill scale bar/text areas
    if len(region.shape) == 3:
        inpainted = cv2.inpaint(region, mask, 3, cv2.INPAINT_TELEA)
    else:
        inpainted = cv2.inpaint(region, mask, 3, cv2.INPAINT_TELEA)
    
    # Put the inpainted region back into the image
    img[y1:h, x1:w] = inpainted
    
    # Save processed image
    cv2.imwrite(out_path, img)
    
    return True


def process_dataset(input_root, output_root):
    if not os.path.exists(input_root):
        print(f"ERROR: Input folder '{input_root}' does not exist!")
        return

    for root, _, files in os.walk(input_root):
        for fname in files:
            if fname.lower().endswith((".tif", ".tiff")):
                in_path = os.path.join(root, fname)

                try:
                    # Preserve folder structure
                    rel_path = os.path.relpath(root, input_root)
                    out_dir = os.path.join(output_root, rel_path)
                    os.makedirs(out_dir, exist_ok=True)

                    out_path = os.path.join(
                        out_dir, f"processed_{fname}"
                    )

                    process_image(in_path, out_path)
                    print(f"{in_path} -> {out_path}")
                    
                except Exception as e:
                    print(f"FAILED: {in_path} - {e}")


if __name__ == "__main__":
    input_root = "./Data"        # MUST exist
    output_root = "./data_processed" # WILL be created
    process_dataset(input_root, output_root)
