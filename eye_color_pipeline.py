import os
from eye_mask import create_pupil_mask # פונקציה שיוצרת מסכת עיניים
from change_eye_color import change_eye_color     # פונקציה שמחליפה צבע עיניים

# ====== SETTINGS ======
INPUT_FOLDER = r"input_faces"
OUTPUT_FOLDER = r"output_faces_change_eye_color"
MASK_FOLDER = r"eye_masks"
NEW_EYE_COLOR = "bright blue eyes"
# ======================

# צור תיקיות אם לא קיימות
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)


def process_all_images():
    print("\n🚀 Starting batch lips & eyes pipeline...\n")

    images = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not images:
        print("No images found in input folder.")
        return

    for img_name in images:
        try:
            print(f"\n🖼 Processing: {img_name}")

            input_path = os.path.join(INPUT_FOLDER, img_name)



            # ===== 1. CREATE EYE MASK =====
            eye_mask_path = os.path.join(MASK_FOLDER, img_name.split('.')[0] + "_eye_mask.png")
            print("Creating eye mask...")
            create_pupil_mask(input_path, eye_mask_path)

            # ===== 2. CHANGE EYE COLOR =====
            final_output_path = os.path.join(OUTPUT_FOLDER, img_name.split('.')[0] + "_lips_eyes_edited.png")
            print("Changing eye color...")
            change_eye_color(input_path, eye_mask_path, NEW_EYE_COLOR, final_output_path)

            print(f"✅ Done: {img_name}")

        except Exception as e:
            print(f"❌ Failed on {img_name}: {e}")


if __name__ == "__main__":
    process_all_images()
