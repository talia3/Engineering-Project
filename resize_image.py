import cv2
import os

# ===== נתיבים =====
folder_path = r"C:\Users\97258\engineering_try_2\Engineering-Project\resize_good_image"
reference_image_path = r"C:\Users\97258\engineering_try_2\Engineering-Project\output_faces_change_eye_color\000044_edited.png"

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

# ===== גודל יעד לפי תמונת רפרנס =====
ref_img = cv2.imread(reference_image_path)
if ref_img is None:
    print("Reference image not found!")
    exit()

target_h, target_w = ref_img.shape[:2]
print(f"Target size: {target_w}x{target_h}")

# תיקיית פלט (לא דורס מקור)
output_folder = os.path.join(folder_path, "resized_with_white_padding")
os.makedirs(output_folder, exist_ok=True)

# ===== פונקציה שמוסיפה שוליים לבנים =====
def resize_with_padding(img, target_w, target_h):
    h, w = img.shape[:2]

    # יחס קנה מידה לשמירת פרופורציות
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # חישוב שוליים
    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    # הוספת שוליים לבנים
    padded = cv2.copyMakeBorder(
        resized,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255]  # לבן
    )

    return padded

# ===== מעבר על כל התמונות =====
images = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

for filename in images:
    img_path = os.path.join(folder_path, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Skipping {filename}")
        continue

    result = resize_with_padding(img, target_w, target_h)

    out_path = os.path.join(output_folder, filename)
    cv2.imwrite(out_path, result)
    print(f"Processed: {filename}")

print("All images resized with white padding 🎉")
