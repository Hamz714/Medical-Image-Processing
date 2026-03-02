import numpy as np
import cv2

def remove_gaussian(img, num_iter=10, kappa=15, gamma=0.1, option=1):
    """
    Applies Perona-Malik Anisotropic Diffusion to an image.
    
    Parameters:
    img      : Input image (grayscale or color).
    num_iter : Number of iterations. More iterations mean more smoothing.
    kappa    : Conduction coefficient. Controls sensitivity to edges. 
               - High kappa: Smooths over most edges (acts like Gaussian blur).
               - Low kappa: Preserves fine details and edges.
    gamma    : Integration constant. Must be <= 0.25 for 2D stability.
    option   : 1 for exponential function (favors high-contrast edges).
               2 for inverse polynomial (favors wide regions over smaller ones).
               
    Returns:
    Anisotropic diffused image.
    """
    # Convert to float32 for precise mathematical operations
    img_out = img.astype(np.float32)
    
    # Process each color channel independently if the image is in color
    if len(img_out.shape) == 3:
        channels = cv2.split(img_out)
    else:
        channels = [img_out]
        
    out_channels = []
    
    for channel in channels:
        out = channel.copy()
        
        for _ in range(num_iter):
            # Pad the image to safely calculate differences at the borders
            padded = np.pad(out, 1, mode='reflect')
            
            # Calculate gradients (differences) in the 4 cardinal directions
            deltaN = padded[:-2, 1:-1] - out
            deltaS = padded[2:, 1:-1] - out
            deltaE = padded[1:-1, 2:] - out
            deltaW = padded[1:-1, :-2] - out
            
            # Calculate conduction gradients (diffusion coefficients)
            if option == 1:
                cN = np.exp(-(deltaN/kappa)**2)
                cS = np.exp(-(deltaS/kappa)**2)
                cE = np.exp(-(deltaE/kappa)**2)
                cW = np.exp(-(deltaW/kappa)**2)
            elif option == 2:
                cN = 1.0 / (1.0 + (deltaN/kappa)**2)
                cS = 1.0 / (1.0 + (deltaS/kappa)**2)
                cE = 1.0 / (1.0 + (deltaE/kappa)**2)
                cW = 1.0 / (1.0 + (deltaW/kappa)**2)
                
            # Update the image by applying the diffusion
            out += gamma * (cN*deltaN + cS*deltaS + cE*deltaE + cW*deltaW)
            
        out_channels.append(out)
        
    # Merge channels back and convert to standard 8-bit image format
    merged = cv2.merge(out_channels) if len(out_channels) > 1 else out_channels[0]
    return np.clip(merged, 0, 255).astype(np.uint8)