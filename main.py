"""
Pre-processing pipeline for X-ray images.

This script implements a pipeline to clean and enhance X-ray images
for classification.
The pipeline performs the following steps in order:
1. Perspective Correction (Dewarping)
2. Inpainting (Removing artifacts/holes)
3. Color Balancing (Gray World Assumption)
4. Noise Removal (Salt & Pepper + Gaussian)
5. Contrast Enhancement (CLAHE)
6. Edge Sharpening

It then evaluates the processed images using a provided ONNX
neural network model.
"""

import os
import argparse
import numpy as np
import cv2


def remove_salt_and_pepper_noise(image, kernel_size=5):
    """
    Removes salt and pepper (impulse) noise using a Median Filter.

    The median filter is highly effective against salt-and-pepper noise because
    it replaces each pixel with the median of its neighbors, effectively
    ignoring outliers (bright white or dark black spots) without blurring
    edges as much as a mean filter would.

    Args:
        image (numpy.ndarray): Input image.
        kernel_size (int): Size of the filter kernel (must be odd).
                           5 is more aggressive, 3 preserves finer details.

    Returns:
        numpy.ndarray: Image with impulse noise removed.
    """
    return cv2.medianBlur(image, kernel_size)


def remove_gaussian_noise(image, h=15, h_color=15):
    """
    Removes Gaussian (static/thermal) noise using Non-Local Means Denoising.

    Unlike simple blurring, Non-Local Means looks for similar patches
    elsewhere in the image to average out noise, preserving structural details
    like bone edges much better than a standard Gaussian blur.

    Args:
        image (numpy.ndarray): Input image.
        h (float): Filter strength for the luminance component. Higher values
                   remove more noise but may smear details.
        h_color (float): Filter strength for color components.

    Returns:
        numpy.ndarray: Image with Gaussian noise removed.
    """
    # 'fastNlMeansDenoisingColored' is used here because the pipeline treats
    # the X-rays as BGR images. It separates luminance and color
    # noise processing.
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=h,
        hColor=h_color,
        templateWindowSize=7,
        searchWindowSize=21
    )


def order_points(pts):
    """
    Orders a list of 4 coordinates in a consistent order:
    top-left, top-right, bottom-right, bottom-left.

    This is required for perspective transforms to map corners correctly.

    Args:
        pts (numpy.ndarray): A list of four (x, y) coordinates.

    Returns:
        numpy.ndarray: Ordered coordinates.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum (x + y),
    # and the bottom-right point will have the largest sum.
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # The top-right point will have the smallest difference (y - x or x - y),
    # and the bottom-left point will have the largest difference.
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def dewarp_image(image):
    """
    Detects the largest contour (the X-ray film) and applies a perspective
    transform to correct skew/warping, returning a flat 256x256 image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Warped and resized 256x256 image.
    """
    # Convert to grayscale for contour detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to separate the dark background from the lighter X-ray film
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    # Find external contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        # Fallback: If no contour is found, just resize the original
        return cv2.resize(image, (256, 256))

    # Assume the largest contour is the X-ray film
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon (simplifying the shape)
    # 0.05 is the approximation accuracy (5% of arc length)
    epsilon = 0.05 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # If the approximated polygon has 4 points, we assume it's a rectangle
    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        rect = order_points(pts)

        # Define destination points for a perfect 256x256 square
        dst = np.array([
            [0, 0],
            [255, 0],
            [255, 255],
            [0, 255]
        ], dtype="float32")

        # Compute the perspective transform matrix and warp
        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, matrix, (256, 256))
        return warped

    # Fallback if 4 corners weren't detected
    return cv2.resize(image, (256, 256))


def enhance_contrast_clahe(image, clip_limit=3.0, grid_size=(8, 8)):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Standard histogram equalization can wash out details if the image has
    large dark/bright areas. CLAHE operates on small tiles (grid_size)
    and limits contrast amplification (clip_limit) to prevent
    noise amplification.

    Args:
        image (numpy.ndarray): Input BGR image.
        clip_limit (float): Threshold for contrast limiting.
        grid_size (tuple): Size of the grid for local histogram equalization.

    Returns:
        numpy.ndarray: Contrast-enhanced image.
    """
    # 1. Convert to LAB color space to access the Lightness channel (L)
    # We only want to enhance luminosity, not shift colors.
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
    Balances color channels using the Gray World assumption.

    This algorithm assumes that the average color of a neutral
    scene should be gray.
    It scales the R, G, and B channels so their means are equal, removing
    color casts (e.g., yellowing or blue tints).

    Args:
        img (numpy.ndarray): Input BGR image.

    Returns:
        numpy.ndarray: Color-balanced image.
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
    k = (b_mean + g_mean + r_mean) / 3.0

    # 4. Scale each channel to match the global goal 'k'.
    # We check for > 0 to avoid division by zero errors.
    if b_mean > 0:
        b = b * (k / b_mean)
    if g_mean > 0:
        g = g * (k / g_mean)
    if r_mean > 0:
        r = r * (k / r_mean)

    # 5. Merge and Clip
    # Values might exceed 255 after scaling,
    # so we must clip them to valid uint8 range.
    balanced = cv2.merge((b, g, r))
    balanced = np.clip(balanced, 0, 255).astype(np.uint8)

    return balanced


def inpaint_missing_region(img):
    """
    Detects pitch-black holes (missing data) and fills them using inpainting.

    Args:
        img (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Image with holes filled.
    """
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Tight Thresholding
    # We use a very low threshold (5) to specifically
    # target pitch-black missing
    # data regions, avoiding dark biological tissue.
    # THRESH_BINARY_INV creates a white mask for the black holes.
    _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY_INV)

    # 3. Filter for the Hole (Largest Contour in the mask)
    # This prevents small dark noise spots from being treated as holes.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    final_mask = np.zeros_like(mask)

    if contours:
        # Find the single largest dark region
        largest_contour = max(contours, key=cv2.contourArea)

        # Draw it onto our final mask
        cv2.drawContours(final_mask, [largest_contour],
                         -1, 255, thickness=cv2.FILLED)

        # 4. Minimal Dilation
        # Expand the mask slightly (1 pixel) to cover the anti-aliased edges
        # of the hole, preventing a "halo" effect after inpainting.
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.dilate(final_mask, kernel, iterations=1)

    # 5. Apply Telea Inpainting
    # Uses pixels from the boundary to interpolate into the missing region.
    inpainted_img = cv2.inpaint(img, final_mask, 3, cv2.INPAINT_TELEA)

    return inpainted_img


def sharpen_edges(image, amount=1.5, threshold=0):
    """
    Sharpens the image using Unsharp Masking.

    Formula: Sharp = Original + (Original - Blurred) * amount

    Args:
        image (numpy.ndarray): Input image.
        amount (float): Strength of the sharpening (1.0 to 2.0 is typical).
        threshold (int): Minimum brightness change required to apply
          sharpening.

    Returns:
        numpy.ndarray: Sharpened image.
    """
    # 1. Create a blurred version of the image (the "Unsharp" mask)
    gaussian_blur = cv2.GaussianBlur(image, (0, 0), 2.0)

    # 2. Calculate the weighted sum to sharpen
    # The formula is effectively: Source * (1+amount) - Blur * amount
    weighted_sharp = cv2.addWeighted(
        image, 1.0 + amount, gaussian_blur, -amount, threshold
    )

    return weighted_sharp


def apply_preprocessing_pipeline(image):
    """
    Master function dictating the order of operations.

    The order is critical:
    1. Dewarp: Normalize geometry first so subsequent
    steps work on the standard view.
    2. Inpaint: Fix missing data before color/contrast
    adjustments distort the hole.
    3. Color Balance: Remove global color casts.
    4. Denoise (S&P then Gaussian): Clean the image before enhancing details.
    5. Enhance Contrast: Bring out features in the clean image.
    6. Sharpen: Finalize edges (done last to avoid sharpening noise).

    Args:
        image (numpy.ndarray): Raw input image.

    Returns:
        numpy.ndarray: Fully processed image.
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
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Run the image processing '
        'pipeline and classify the results.'
    )

    # CHANGED: 'data' is now a positional argument (no '--' prefix required)
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

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: '{output_dir}'")

    # Load the Neural Network Model
    try:
        model = cv2.dnn.readNetFromONNX(args.model)
        print("Successfully loaded the classifier model.")
    except Exception as e:
        print(f"Error loading model '{args.model}': {e}")
        return

    # Define ground truth labels based on filename conventions
    # Images 1-50 are Healthy, 51-100 are Pneumonia
    healthys = tuple(f'im{str(i).zfill(3)}' for i in range(1, 51))
    pneumonias = tuple(f'im{str(i).zfill(3)}' for i in range(51, 101))

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    # Get list of files
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

    # Sort filenames to ensure consistent processing order
    filenames.sort()

    # Mac OS specific: remove hidden .DS_Store file if present
    if ".DS_Store" in filenames:
        filenames.remove(".DS_Store")

    print(f"Found {len(filenames)} images. Beginning processing...\n")

    correct_predictions = 0

    # Main Processing Loop
    for i, filename in enumerate(filenames):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        img = cv2.imread(input_path)

        if img is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            continue

        # --- RUN PIPELINE ---
        processed_img = apply_preprocessing_pipeline(img)

        # Save the result
        cv2.imwrite(output_path, processed_img)

        # --- EVALUATION ---
        # Prepare image for the Neural Network (resize, scaling, swapRB)
        blob = cv2.dnn.blobFromImage(
            processed_img, 1.0 / 255, (256, 256),
            (0, 0, 0), swapRB=True, crop=False
        )
        model.setInput(blob)
        output = model.forward()

        # Handle different output shapes from the model
        prediction_val = output[0][0] if isinstance(output, list) \
            or len(output.shape) > 1 else output

        # Check prediction against Ground Truth
        # Prediction > 0.5 implies Pneumonia (positive class)
        if prediction_val > 0.5:
            # print(f'{filename}: pneumonia')
            if filename.startswith(pneumonias):
                correct_predictions += 1
        else:
            # print(f'{filename}: healthy')
            if filename.startswith(healthys):
                correct_predictions += 1

        print(f"[{i+1}/{len(filenames)}] Processed & Evaluated: {filename}")

    # Final Statistics
    if len(filenames) > 0:
        accuracy = (correct_predictions / len(filenames)) * 100
        print("-" * 50)
        print("PIPELINE COMPLETE")
        print("-" * 50)
        print(f"Total Correct Predictions: {correct_predictions} \
               / {len(filenames)}")
        print(f"Final Classifier Accuracy: {accuracy:.2f}%")
        print("-" * 50)
    else:
        print("No files processed.")


if __name__ == "__main__":
    main()
