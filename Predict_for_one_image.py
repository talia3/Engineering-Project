# Predict_for_one_image.py
import cv2
import numpy as np
import joblib
import os
from skimage.feature import hog

from lips_pipeline import process_single_image_lips
from batch_eye_color_pipeline import process_single_image_eyes

# =========================
# Settings
# =========================
CATEGORIES = [
    "No_Manipulation",
    "output_faces_change_lip_color_no_padding",
    "output_faces_change_eye_color_no_padding"
]

IMG_SIZE = 128
TARGET_SIZE = 512  # size for padded manipulation
REFERENCE_IMAGE_PATH_RESIZE = r"C:\Users\97258\engineering_try_2\Engineering-Project\tryIn\000001.jpg"
REFERENCE_IMAGE_PATH_PAD = r"C:\Users\97258\engineering_try_2\Engineering-Project\output_faces_change_lip_color_with_padding\600_lips_edited.png"
# Load trained SVM model
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# HOG extraction
# =========================
def get_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray, orientations=9, pixels_per_cell=(8,8),
                   cells_per_block=(2,2), block_norm='L2-Hys')
    return features

# =========================
# Resize and padding
# =========================

def resize_to_reference(img, reference_path):
    ref_img = cv2.imread(reference_path)
    if ref_img is None:
        raise ValueError("Reference image not found")

    target_h, target_w = ref_img.shape[:2]

    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    return resized

def pad_to_reference_size(img, reference_path):
    ref_img = cv2.imread(reference_path)
    if ref_img is None:
        raise ValueError("Reference image not found")

    target_h, target_w = ref_img.shape[:2]
    h, w = img.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    delta_w = target_w - new_w
    delta_h = target_h - new_h

    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    padded = cv2.copyMakeBorder(
        resized,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )

    return padded

# =========================
# Remove padding
# =========================
def remove_padding_simple(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    non_white = gray < 245
    rows, cols = np.any(non_white, axis=1), np.any(non_white, axis=0)
    if rows.any() and cols.any():
        y_min, y_max = np.where(rows)[0][[0,-1]]
        x_min, x_max = np.where(cols)[0][[0,-1]]
        return image[y_min:y_max, x_min:x_max]
    return image


# =========================
# Run manipulation pipeline
# =========================
def run_manipulation_pipeline(manipulation, image_path):
    if manipulation == "output_faces_change_lip_color_no_padding":
        return process_single_image_lips(image_path)
    elif manipulation == "output_faces_change_eye_color_no_padding":
        return process_single_image_eyes(image_path)
    else:
        return image_path

# =========================
# Prediction
# =========================
def predict_image(image):
    image_resized = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
    features = get_hog_features(image_resized)
    features = np.array(features).reshape(1,-1)
    features = scaler.transform(features)
    prediction = model.predict(features)
    return CATEGORIES[prediction[0]]

# =========================
# Full pipeline
# =========================
def run_full_pipeline(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found!")
        return
    
    resizes_image = resize_to_reference(image,REFERENCE_IMAGE_PATH_RESIZE)
    padded = pad_to_reference_size(resizes_image, REFERENCE_IMAGE_PATH_PAD)
    padded_path ='padded_image.jpg'
    cv2.imwrite(padded_path, padded)
    print("padded image saved to", padded_path)


    print("Choose manipulation:")
    for i,c in enumerate(CATEGORIES):
        print(i,c)
    choice = int(input("Enter number: "))
    manipulation = CATEGORIES[choice]

    manipulated_path = run_manipulation_pipeline(manipulation, padded_path)
    final_img = cv2.imread(manipulated_path)
    final_img = remove_padding_simple(final_img)

    result = predict_image(final_img)
    print("Prediction:", result)

# =========================
# Run
# =========================
if __name__ == "__main__":
    IMAGE_PATH = r"C:\Users\97258\engineering_try_2\Engineering-Project\178.jpg"
    run_full_pipeline(IMAGE_PATH)
    