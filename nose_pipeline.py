import os
from nose_mask2701 import create_nose_mask  # פונקציה שיוצרת מסכה
from change_nose import change_nose    # פונקציה שמשנה צבע עיניים

# ====== SETTINGS ======
INPUT_FOLDER = r"input_faces"
OUTPUT_FOLDER = r"output_faces_change_nose"
MASK_FOLDER = r"nose_masks"
# ======================

# צור תיקיות אם לא קיימות
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)



def process_all_images():
    print("\n🚀 Starting nose pipeline...\n")

    images = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not images:
        print("No images found in input folder.")
        return

    for img_name in images:
        try:
            print(f"\n🖼 Processing: {img_name}")

            input_path = os.path.join(INPUT_FOLDER, img_name)

            # ===== 1. CREATE MASK =====
            mask_output_path = os.path.join(MASK_FOLDER, img_name.split('.')[0] + "_mask.png")
            print("Creating nose mask...")
            create_nose_mask(input_path, mask_output_path)

            # ===== 2. CHANGE NOSE COLOR =====
            edited_output_path = os.path.join(OUTPUT_FOLDER, img_name.split('.')[0] + "_edited.png")
            print("Changing nose...")
            change_nose(input_path, mask_output_path, edited_output_path)

            print(f"✅ Done: {img_name}")

        except Exception as e:
            print(f"❌ Failed on {img_name}: {e}")


if __name__ == "__main__":
    process_all_images()
