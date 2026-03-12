import os
import cv2
from skimage.feature import hog
from skimage import exposure

# Settings
INPUT_DIR = r"C:\Users\97258\engineering_try_2\Engineering-Project\HOG_Visualizations\run_hog"
OUTPUT_DIR = r"C:\Users\97258\engineering_try_2\Engineering-Project\HOG_Visualizations"

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def save_hog_visualization(img_path, output_path):
    # 1. Load image
    image = cv2.imread(img_path)
    if image is None:
        return
    
    # 2. Resize image to 845x1024
    image = cv2.resize(image, (845, 1024))

    # 3. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 4. Run HOG with visualization enabled
    fd, hog_image = hog(
        gray, 
        orientations=9, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        block_norm='L2-Hys',
        visualize=True
    )

    # 5. Rescale intensity for better visibility
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    # 6. Convert to 8-bit image
    hog_image_8bit = (hog_image_rescaled * 255).astype("uint8")
    
    # 7. Save the result
    cv2.imwrite(output_path, hog_image_8bit)

# Process a few images as a test
for img_name in os.listdir(INPUT_DIR)[:5]:
    input_p = os.path.join(INPUT_DIR, img_name)
    output_p = os.path.join(OUTPUT_DIR, f"hog_{img_name}")
    save_hog_visualization(input_p, output_p)
    print(f"Saved: {output_p}")