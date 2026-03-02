"""
Pre-processing pipeline for X-ray images.
Removes mixed noise, dynamically corrects perspective warping, 
balances colour channels, and enhances contrast using CLAHE.
Evaluates the processed images using the provided neural network model.
Complies with PEP8 styling guidelines.
"""

import os
import argparse
import numpy as np
import cv2


import cv2

def remove_salt_and_pepper_noise(image, kernel_size=5):
    """
    Removes salt and pepper noise using a Median Filter.
    
    Args:
        image (numpy.ndarray): Input image.
        kernel_size (int): Size of the filter (3 or 5). 
                           5 is more aggressive, 3 is sharper.
    Returns:
        numpy.ndarray: Image with impulse noise removed.
    """
    # Median filter is the gold standard for S&P noise
    return cv2.medianBlur(image, kernel_size)


def remove_gaussian_noise(image, h=15, hColor=15):
    """
    Removes Gaussian (static) noise using Non-Local Means Denoising.
    
    Args:
        image (numpy.ndarray): Input image.
        h (float): Filter strength for luminance component.
        hColor (float): Filter strength for color component.
        
    Returns:
        numpy.ndarray: Image with Gaussian noise removed.
    """
    # 'fastNlMeansDenoisingColored' works on BGR images and preserves edges
    # better than standard Gaussian Blur.
    return cv2.fastNlMeansDenoisingColored(image, None, 
                                           h=h, hColor=hColor, 
                                           templateWindowSize=7, 
                                           searchWindowSize=21)


def order_points(pts):
    """
    Orders 4 coordinates in the following order:
    top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # The top-left point will have the smallest sum,
    # and the bottom-right point will have the largest sum.
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # The top-right point will have the smallest difference,
    # and the bottom-left point will have the largest difference.
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def dewarp_image(image):
    """
    Dynamically finds the skewed X-ray contour and applies a 
    perspective transform to warp it back to a flat 256x256 image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        # Fallback to returning a 256x256 resized image
        return cv2.resize(image, (256, 256))
        
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.05 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        rect = order_points(pts)
        
        # Define destination points for a flat 256x256 image
        dst = np.array([
            [0, 0],
            [255, 0],
            [255, 255],
            [0, 255]
        ], dtype="float32")
        
        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, matrix, (256, 256))
        return warped
        
    return cv2.resize(image, (256, 256))


def enhance_contrast_clahe(image, clip_limit=3.0, grid_size=(8, 8)):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to 
    enhance local details (bones/tissue) that might look flat.
    
    Args:
        image (numpy.ndarray): Input BGR image.
        clip_limit (float): Threshold for contrast limiting.
        grid_size (tuple): Size of the grid for local histogram equalization.
        
    Returns:
        numpy.ndarray: Contrast-enhanced image.
    """
    # 1. Convert to LAB color space to access the Lightness channel (L)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 2. Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_enhanced = clahe.apply(l)
    
    # 3. Merge channels and convert back to BGR
    lab_merged = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)


def fix_color_imbalance_gray_world(img):
    """
    Balances the color channels using the Gray World assumption.
    Scales R, G, and B so their averages are equal (removing color casts).
    """
    # Convert to float32 to prevent overflow during multiplication
    img_float = img.astype(np.float32)
    
    # 1. Split the channels
    b, g, r = cv2.split(img_float)
    
    # 2. Calculate the average brightness of each channel
    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)
    
    # 3. Calculate the "Global Goal" (The average gray value we want)
    # This is the "Gray" world average
    k = (b_mean + g_mean + r_mean) / 3.0
    
    # 4. Scale each channel to match the global goal 'k'
    # If a channel is too dark (mean < k), this multiplier will be > 1.0 (brightening it)
    # If a channel is too bright (mean > k), this multiplier will be < 1.0 (darkening it)
    # We add a tiny epsilon (1e-5) to avoid division by zero errors
    if b_mean > 0: b = b * (k / b_mean)
    if g_mean > 0: g = g * (k / g_mean)
    if r_mean > 0: r = r * (k / r_mean)
    
    # 5. Merge and Clip
    # Values might exceed 255 after scaling, so we must clip them
    balanced = cv2.merge((b, g, r))
    balanced = np.clip(balanced, 0, 255).astype(np.uint8)
    
    return balanced


def inpaint_missing_region(img):
    """
    Precise detection of the missing region without over-masking.
    """
    # 1. Convert to Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Tighter Threshold
    # We lower the threshold to 5. This ensures we only catch the pitch-black hole
    # and NOT the dark gray lung tissue.
    _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Filter for the Hole (Largest Contour)
    # This removes small "pepper" noise without needing to blur/dilate the whole mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_mask = np.zeros_like(mask)
    
    if contours:
        # Find the single largest dark region (the hole)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Draw it exactly as is (no extra growth)
        cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # 4. Minimal Dilation
        # We grow the mask by just 1 pixel to cover the jagged anti-aliasing edge.
        # This prevents the "white halo" effect around the filled hole.
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.dilate(final_mask, kernel, iterations=1)
    
    # 5. Apply Inpainting
    # radius=3 is small enough to keep texture but large enough to heal the edge.
    inpainted_img = cv2.inpaint(img, final_mask, 3, cv2.INPAINT_TELEA)
    
    return inpainted_img


def sharpen_edges(image, amount=1.5, threshold=0):
    """
    Sharpens the image using Unsharp Masking technique.
    Formula: Sharp = Original + (Original - Blurred) * amount
    
    Args:
        image (numpy.ndarray): Input image.
        amount (float): Strength of the sharpening (1.0 to 2.0).
        threshold (int): Minimum brightness change to be sharpened (0=all).
        
    Returns:
        numpy.ndarray: Sharpened image.
    """
    # 1. Create a blurred version of the image
    gaussian_blur = cv2.GaussianBlur(image, (0, 0), 2.0)
    
    # 2. Calculate the weighted sum to sharpen
    # weight_original = 1 + amount
    # weight_blur = -amount
    weighted_sharp = cv2.addWeighted(image, 1.0 + amount, gaussian_blur, -amount, threshold)
    
    return weighted_sharp


def apply_preprocessing_pipeline(image):
    """
    Master function dictating the order of operations.
    Order: Median Blur -> Dewarp -> Contrast/Color -> NLM Denoise.
    """
    image = dewarp_image(image)
    image = inpaint_missing_region(image)
    image = fix_color_imbalance_gray_world(image)
    image = remove_salt_and_pepper_noise(image)
    image = remove_gaussian_noise(image)
    image = enhance_contrast_clahe(image)
    image = sharpen_edges(image)
    
    return image


def main():
    parser = argparse.ArgumentParser(
        description='Run the image processing pipeline and classify the results.'
    )

    parser.add_argument(
        "data", 
        type=str,
        help="Path to the noisy input images directory"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Specify path to model weights",
        default='classifier.model'
    )

    args = parser.parse_args()

    input_dir = args.data
    output_dir = 'Results'  

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: '{output_dir}'")

    try:
        model = cv2.dnn.readNetFromONNX(args.model)
        print("Successfully loaded the classifier model.")
    except Exception as e:
        print(f"Error loading model '{args.model}': {e}")
        return

    healthys = tuple(f'im{str(i).zfill(3)}' for i in range(1, 51))
    pneumonias = tuple(f'im{str(i).zfill(3)}' for i in range(51, 101))

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    try:
        filenames = [
            f for f in os.listdir(input_dir) 
            if f.lower().endswith(valid_extensions)
        ]
    except FileNotFoundError:
        print(f"Error: The directory '{input_dir}' was not found.")
        return

    if not filenames:
        print(f"No images found in '{input_dir}'.")
        return

    filenames.sort()
    if ".DS_Store" in filenames:
        filenames.remove(".DS_Store")

    print(f"Found {len(filenames)} images. Beginning processing...\n")
    
    correct_predictions = 0

    for i, filename in enumerate(filenames):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        img = cv2.imread(input_path)

        if img is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            continue

        processed_img = apply_preprocessing_pipeline(img)

        cv2.imwrite(output_path, processed_img)
        
        blob = cv2.dnn.blobFromImage(
            processed_img, 1.0 / 255, (256, 256), 
            (0, 0, 0), swapRB=True, crop=False
        )
        model.setInput(blob)
        output = model.forward()

        prediction_val = output[0][0] if isinstance(output, list) or len(output.shape) > 1 else output
        
        if prediction_val > 0.5:
            if filename.startswith(pneumonias):
                correct_predictions += 1
        else:
            if filename.startswith(healthys):
                correct_predictions += 1

        print(f"[{i+1}/{len(filenames)}] Processed & Evaluated: {filename}")

    accuracy = (correct_predictions / len(filenames)) * 100
    print("-" * 50)
    print("PIPELINE COMPLETE")
    print("-" * 50)
    print(f"Total Correct Predictions: {correct_predictions} / {len(filenames)}")
    print(f"Final Classifier Accuracy: {accuracy:.2f}%")
    print("-" * 50)


if __name__ == "__main__":
    main()