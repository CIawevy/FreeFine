import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# def color_mask(mask_path, mask2_path=None, mask_color=(255, 215, 0), mask2_color=(0, 255, 0), non_mask_color=(0, 51, 102)):
#     """
#     Generate a colored mask from one or two grayscale mask images.
#
#     Parameters:
#     - mask_path: Path to the first mask image (grayscale image where mask region is typically white).
#     - mask2_path: Path to the second mask image (optional).
#     - mask_color: RGB tuple for the first mask region (default is yellow, #FFD700).
#     - mask2_color: RGB tuple for the second mask region (default is green, #00FF00).
#     - non_mask_color: RGB tuple for the non-mask region (default is dark blue, #003366).
#
#     Returns:
#     - colored_mask: The colored mask image as a NumPy array.
#     """
#     # Open the first mask image
#     mask = Image.open(mask_path).convert('L')
#     mask_array = np.array(mask)
#
#     # Create a colored output image
#     colored_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
#
#     # Set non-mask region color
#     colored_mask[mask_array == 0] = non_mask_color
#
#     # Set first mask region color
#     colored_mask[mask_array > 0] = mask_color
#
#     # If a second mask is provided, overlay it
#     if mask2_path is not None:
#         mask2 = Image.open(mask2_path).convert('L')
#         mask2_array = np.array(mask2)
#
#         # Ensure the second mask has the same dimensions as the first mask
#         if mask2_array.shape != mask_array.shape:
#             mask2_array = np.array(Image.fromarray(mask2_array).resize((mask_array.shape[1], mask_array.shape[0]), Image.Resampling.LANCZOS))
#
#         # Set second mask region color
#         colored_mask[mask2_array > 0] = mask2_color
#
#     return colored_mask
def color_mask(mask_path, mask2_path=None, mask_color=(255, 215, 0), mask2_color=(0, 255, 0), non_mask_color=(0, 51, 102)):
    """
    Generate a colored mask from one or two grayscale mask images.

    Parameters:
    - mask_path: Path to the first mask image (grayscale image where mask region is typically white).
    - mask2_path: Path to the second mask image (optional).
    - mask_color: RGB tuple for the first mask region (default is yellow, #FFD700).
    - mask2_color: RGB tuple for the second mask region (default is green, #00FF00).
    - non_mask_color: RGB tuple for the non-mask region (default is dark blue, #003366).

    Returns:
    - colored_mask: The colored mask image as a NumPy array.
    """
    # Open the first mask image
    mask = Image.open(mask_path).convert('L')
    mask2 = Image.open(mask2_path).convert('L')
    mask_array = np.array(mask)
    mask_array[mask_array>0] = 1
    mask2_array = np.array(mask2)
    mask2_array[mask2_array>0] = 1
    mask_full_array = mask_array + mask2_array
    mask_full_array[mask_full_array>0] = 1
    mask_full_array = mask_full_array.astype(np.uint8)*255
    # Create a colored output image
    colored_mask = np.zeros((mask_full_array.shape[0], mask_full_array.shape[1], 3), dtype=np.uint8)

    # Set non-mask region color
    colored_mask[mask_full_array == 0] = non_mask_color

    # Set first mask region color
    colored_mask[mask_full_array > 0] = mask_color


    return colored_mask

def blend_image_with_mask(img_path, mask_path, output_path, mask2_path=None, alpha=0.5, mask_color=(255, 215, 0), mask2_color=(0, 255, 0), non_mask_color=(0, 51, 102)):
    """
    Blend the original image with one or two colored masks.

    Parameters:
    - img_path: Path to the original image.
    - mask_path: Path to the first mask image.
    - mask2_path: Path to the second mask image (optional).
    - output_path: Path to save the blended image.
    - alpha: Blending factor (0 = fully original image, 1 = fully colored mask).
    - mask_color: RGB tuple for the first mask region (default is yellow, #FFD700).
    - mask2_color: RGB tuple for the second mask region (default is green, #00FF00).
    - non_mask_color: RGB tuple for the non-mask region (default is dark blue, #003366).
    """
    # Load the original image
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)

    # Generate the colored mask
    colored_mask = color_mask(mask_path, mask2_path, mask_color, mask2_color, non_mask_color)

    # Resize the mask to match the image dimensions
    if colored_mask.shape[:2] != img_array.shape[:2]:
        print("Resizing mask to match image dimensions...")
        colored_mask = Image.fromarray(colored_mask).resize((img_array.shape[1], img_array.shape[0]), Image.Resampling.LANCZOS)
        colored_mask = np.array(colored_mask)

    # Blend the image and mask
    blended_array = (img_array * (1 - alpha) + colored_mask * alpha).astype(np.uint8)

    # Save the blended image
    blended_img = Image.fromarray(blended_array)
    blended_img.save(output_path)

    print(f"Blended image saved at: {output_path}")

    # Display the blended image
    plt.imshow(blended_array)
    plt.axis('off')  # Hide axes for better view
    plt.show()
def color_mask_v2(img_path, mask_path, output_path, mask2_path=None, alpha=0.5, mask_color=(255, 215, 0), mask2_color=(0, 255, 0), non_mask_color=(0, 51, 102)):
    """
    Blend the original image with one or two colored masks.

    Parameters:
    - img_path: Path to the original image.
    - mask_path: Path to the first mask image.
    - mask2_path: Path to the second mask image (optional).
    - output_path: Path to save the blended image.
    - alpha: Blending factor (0 = fully original image, 1 = fully colored mask).
    - mask_color: RGB tuple for the first mask region (default is yellow, #FFD700).
    - mask2_color: RGB tuple for the second mask region (default is green, #00FF00).
    - non_mask_color: RGB tuple for the non-mask region (default is dark blue, #003366).
    """
    # Load the original image
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)

    # Generate the colored mask
    colored_mask = color_mask(mask_path, mask2_path, mask_color, mask2_color, non_mask_color)

    # Resize the mask to match the image dimensions
    if colored_mask.shape[:2] != img_array.shape[:2]:
        print("Resizing mask to match image dimensions...")
        colored_mask = Image.fromarray(colored_mask).resize((img_array.shape[1], img_array.shape[0]), Image.Resampling.LANCZOS)
        colored_mask = np.array(colored_mask)

    # # Blend the image and mask
    # blended_array = (img_array * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
    blended_array = colored_mask
    # Save the blended image
    blended_img = Image.fromarray(blended_array)
    blended_img.save(output_path)

    print(f"Blended image saved at: {output_path}")

    # Display the blended image
    plt.imshow(blended_array)
    plt.axis('off')  # Hide axes for better view
    plt.show()
# Example usage
mask_color = (255, 215, 0)  # Yellow for the first mask
mask2_color = (189, 197, 222)   # Green for the second mask
non_mask_color = (0, 51, 102)  # Dark blue for the background

save_path = "/data/Hszhu/dataset/Geo-Bench-SC/mask_img.png"
mask_path = "/data/Hszhu/dataset/Geo-Bench-SC/source_mask/23/0.png"
# mask_path = "/data/Hszhu/dataset/Geo-Bench-SC/target_mask/23/0/0.png"
mask2_path = "/data/Hszhu/dataset/Geo-Bench-SC/draw_mask/23/0/draw_0.png"  # Path to the second mask
img_path = "/data/Hszhu/dataset/Geo-Bench-SC/source_img/23.png"
img_path = "/data/Hszhu/dataset/Geo-Bench-SC/coarse_img/23/0/0.png"

# Blend the image with the masks (alpha controls the blending strength)
# blend_image_with_mask(img_path, mask_path, save_path, mask2_path, alpha=0.7, mask_color=mask_color, mask2_color=mask2_color, non_mask_color=non_mask_color)
color_mask_v2(img_path, mask_path, save_path, mask2_path, alpha=0.7, mask_color=mask_color, mask2_color=mask2_color, non_mask_color=non_mask_color)
# color_mask (mask_path, save_path, mask2_path, mask_color=mask_color, mask2_color=mask2_color, non_mask_color=non_mask_color)