import albumentations as A
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import json
from PIL import Image



file_path = 'C:/Users/jojor/Desktop/Uni/Master/2425WS/DSSS/HW6/rectangles_dsss.sec'



def extract_rectangles(group_path):
    """
    Extract rectangle data from the specified group in the HDF5 file.
    Handles cases where data is stored in attributes.
    """
    rectangles = []
    with h5py.File(file_path, 'r') as file:
        group = file[group_path]
        for key in group:
            obj = group[key]
            if isinstance(obj, h5py.Group):
                # Extract rectangle data from attributes
                attrs = obj.attrs
                if all(attr in attrs for attr in ['i0', 'i1', 'i2', 'i3']):
                    rectangle = (
                        float(attrs['i0']),
                        float(attrs['i1']),
                        float(attrs['i2']),
                        float(attrs['i3'])
                    )
                    rectangles.append(rectangle)
                else:
                    raise ValueError(f"Missing expected attributes in {group_path}/{key}")
            else:
                raise ValueError(f"Unexpected structure at {group_path}/{key}")
    return rectangles



def calculate_iou(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    # intersection coordinates
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    # intersection area
    intersection_width = max(0, xi2 - xi1)
    intersection_height = max(0, yi2 - yi1)
    intersection_area = intersection_width * intersection_height
    # union area
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area

    if union_area == 0:
        return 0
    return intersection_area / union_area

def plot_iou_scores(iou_scores):
    bins = np.linspace(0, 1, 11)  # Define bins from 0 to 1
    histogram, bin_edges = np.histogram(iou_scores, bins=bins)

    for i in range(len(histogram)):
        print(f"Range {bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}: {histogram[i]}")

    plt.bar(bin_edges[:-1], histogram, width=0.1, align='edge', color='blue', edgecolor='black')
    plt.xlabel('IoU Score')
    plt.ylabel('Frequency')
    plt.title('IoU Score Distribution')
    plt.show()

# Main script
# Extract ground truth and predicted rectangles
ground_truth_rectangles = extract_rectangles('ground_truth')
predicted_rectangles = extract_rectangles('predicted')

# Calculate IoU scores
iou_scores = [
    calculate_iou(gt, pred)
    for gt, pred in zip(ground_truth_rectangles, predicted_rectangles)
]

# Plot the IoU scores
plot_iou_scores(iou_scores)


## Task 2
matriculation_number = 22790401
np.random.seed(matriculation_number)


def load_data(dataset_folder, num_images=1):
    # Randomly select .meta files
    selected_files = random.sample(
        [f for f in os.listdir(dataset_folder) if f.endswith('.meta')], num_images
    )

    data = []

    for file in selected_files:
        base_name = file.split('.')[0]  # e.g., "0" from "0.meta"
        image_path = os.path.join(dataset_folder, f"{base_name}.png")
        mask_path = os.path.join(dataset_folder, f"{base_name}_seg.png")

        try:
            # Load the image and mask
            image = Image.open(image_path)
            mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

            # Convert PIL image to numpy array for augmentation
            image_np = np.array(image)
            mask_np = np.array(mask)

            data.append((image_np, mask_np))

        except Exception as e:
            print(f"Error loading data for {base_name}: {e}")

    return data


# Load data from dataset folder
dataset_folder = "C:/Users/jojor/Desktop/Uni/Master/2425WS/DSSS/HW6/Mini_BAGLS_dataset"
loaded_data = load_data(dataset_folder)

# Albumentations augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=45, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.Resize(256, 256),  # Resize images to a consistent size
])

# Select the first image and mask
image, mask = loaded_data[0]

# Apply transformations
augmented_1 = transform(image=image, mask=mask)
augmented_2 = transform(image=image, mask=mask)
augmented_3 = transform(image=image, mask=mask)
augmented_4 = transform(image=image, mask=mask)

# Prepare the augmented images and masks
augmented_images = [
    augmented_1['image'],
    augmented_2['image'],
    augmented_3['image'],
    augmented_4['image']
]

augmented_masks = [
    augmented_1['mask'],
    augmented_2['mask'],
    augmented_3['mask'],
    augmented_4['mask']
]

# Use plt.subplots for flexible grid layout
fig, axs = plt.subplots(5, 2, figsize=(12, 20))  # 5 rows, 2 columns

# Original image and mask
axs[0, 0].imshow(image)
axs[0, 0].set_title("Original Image")
axs[0, 0].axis("off")

axs[0, 1].imshow(mask, cmap='gray')
axs[0, 1].set_title("Original Mask")
axs[0, 1].axis("off")

# Augmented images and masks
for i in range(4):
    augmented_image = augmented_images[i]
    augmented_mask = augmented_masks[i]

    # Create a title indicating the applied augmentation
    if i == 0:
        title = "Augmented with Horizontal Flip"
    elif i == 1:
        title = "Augmented with Vertical Flip"
    elif i == 2:
        title = "Augmented with Random Brightness"
    else:
        title = "Augmented with Rotate and Shift"

    # Plot augmented image and mask in the respective grid positions
    axs[i + 1, 0].imshow(augmented_image)
    axs[i + 1, 0].set_title(title)
    axs[i + 1, 0].axis("off")

    axs[i + 1, 1].imshow(augmented_mask, cmap='gray')
    axs[i + 1, 1].axis("off")

# Adjust layout for better spacing
plt.tight_layout()
plt.show()
