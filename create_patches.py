import cv2
import numpy as np
from PIL import Image, ImageDraw
import json
import matplotlib.pyplot as plt
import argparse
import os

def detect_circle(image_path):
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(image, (9, 9), 2)
    
    # Use the HoughCircles function to detect circles in the image
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=100, param2=30, minRadius=0, maxRadius=0)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            return x, y, r
    return None

def extract_patches(image_path, center, radius, patch_size, json_path):
    image = Image.open(image_path)
    width, height = image.size
    patches = []
    patch_id = 0

    x_center, y_center = center
    for x in range(x_center - radius, x_center + radius, patch_size):
        for y in range(y_center - radius, y_center + radius, patch_size):
            if x + patch_size > width or y + patch_size > height:
                continue  # Skip patches that would exceed image boundaries

            # Check if the center of the patch is within the circle
            x_mid = x + patch_size // 2
            y_mid = y + patch_size // 2
            if (x_mid - x_center) ** 2 + (y_mid - y_center) ** 2 > radius ** 2:
                continue  # Skip patches outside the circle

            x_right = x + patch_size
            y_low = y + patch_size
            y_up = y
            x_left = x

            # Save coordinates
            patch_info = {
                "patch_id": patch_id,
                "x_left": x_left,
                "x_right": x_right,
                "y_low": y_low,
                "y_up": y_up
            }
            patches.append(patch_info)
            patch_id += 1

    # Save coordinates to JSON
    with open(json_path, 'w') as json_file:
        json.dump(patches, json_file, indent=4)

    print(f"Generated {patch_id} patches and saved coordinates to {json_path}")

def visualize_patch(image_path, json_path, patch_id):
    # Load the image
    image = Image.open(image_path)

    # Load the JSON file
    with open(json_path, 'r') as json_file:
        patches = json.load(json_file)

    # Find the patch with the specified patch_id
    patch_info = None
    for patch in patches:
        if patch['patch_id'] == patch_id:
            patch_info = patch
            break

    if patch_info is None:
        print(f"Patch with id {patch_id} not found.")
        return

    # Extract the coordinates
    x_left = patch_info['x_left']
    x_right = patch_info['x_right']
    y_up = patch_info['y_up']
    y_low = patch_info['y_low']

    # Crop the patch from the image
    patch = image.crop((x_left, y_up, x_right, y_low))

    # Display the patch
    plt.imshow(patch)
    plt.title(f'Patch ID: {patch_id}')
    plt.axis('off')
    plt.show()

def draw_patches_on_image(image_path, json_path, output_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Load the JSON file
    with open(json_path, 'r') as json_file:
        patches = json.load(json_file)

    # Draw rectangles on the image
    for patch in patches:
        x_left = patch['x_left']
        x_right = patch['x_right']
        y_up = patch['y_up']
        y_low = patch['y_low']

        draw.rectangle([x_left, y_up, x_right, y_low], outline="red", width=2)

    image.save(output_path)
    
def save_patches_as_png(image_path, json_path, output_dir):
    # /home/yec23006/projects/research/oc/testimg/G1 Ano_3ng RANKL_stack_adj_B.jpg
    image = Image.open(image_path)
    cy5 = Image.open(os.path.join(os.path.dirname(image_path), 'G1 Ano_3ng RANKL_stack_adj_B.jpg')) # Blue
    elf97 = Image.open(os.path.join(os.path.dirname(image_path), 'G1 Ano_3ng RANKL_stack_adj_Y.jpg')) # Yellow
    fitc = Image.open(os.path.join(os.path.dirname(image_path), 'G1 Ano_3ng RANKL_stack_adj_G.jpg')) # Green

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the JSON file
    with open(json_path, 'r') as json_file:
        patches = json.load(json_file)

    # Extract and save patches
    for patch in patches:
        x_left = patch['x_left']
        x_right = patch['x_right']
        y_up = patch['y_up']
        y_low = patch['y_low']

		# merged
        patch_image = image.crop((x_left, y_up, x_right, y_low))
        patch_filename = os.path.join(output_dir, 'A', f"{patch['patch_id']}.png")
        patch_image.save(patch_filename)
        # cy5
        patch_cy5 = cy5.crop((x_left, y_up, x_right, y_low))
        patch_filename = os.path.join(output_dir, 'B', f"{patch['patch_id']}.png")
        patch_cy5.save(patch_filename)
        # elf97
        patch_elf97 = elf97.crop((x_left, y_up, x_right, y_low))
        patch_filename = os.path.join(output_dir, 'G', f"{patch['patch_id']}.png")
        patch_elf97.save(patch_filename)
        # fitc
        patch_fitc = fitc.crop((x_left, y_up, x_right, y_low))
        patch_filename = os.path.join(output_dir, 'Y', f"{patch['patch_id']}.png")
        patch_fitc.save(patch_filename)

    print(f"Saved {len(patches)} patches to {output_dir}")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Extract patches from a WSI and save their coordinates to a JSON file.")
    parser.add_argument("--path2elf97", type=str, default='/home/yec23006/projects/research/oc/testimg/G1 Ano_3ng RANKL_stack_adj_Y.jpg', help="Path to the WSI image file (PNG or JPG).")
    parser.add_argument("--path2img", type=str, default='/home/yec23006/projects/research/oc/testimg/G1 Ano_3ng RANKL_stack_adj_A.jpg', help="Path to the WSI image file (PNG or JPG).")
    parser.add_argument("--patch_size", type=int, default=128, help="Size of each patch (e.g., 64 for 64x64 pixels).")
    parser.add_argument("--output_dir", type=str, default="/home/yec23006/projects/research/oc/patches_dpi/", help="Directory to save the extracted patch images.")
    args = parser.parse_args()
    path2json = os.path.join(os.path.dirname(args.path2img), 'patches_coordinates.json')
    
    center = detect_circle(args.path2elf97)
    print('Detecting A Circle')
    patchOnImg = os.path.join(os.path.dirname(args.path2img), 'patches_on_wsi.png')

    print('Patches Extracting')
    if center is not None:
        x_center, y_center, radius = center
        extract_patches(args.path2img, (x_center, y_center), radius, args.patch_size, path2json)
        draw_patches_on_image(args.path2img, path2json, patchOnImg)
        save_patches_as_png(args.path2img, path2json, args.output_dir)
    else:
        print("No circle detected.")

