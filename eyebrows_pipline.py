import os
from eyebrow_mask import create_eyebrows_mask  # פונקציה שיוצרת מסכה
from change_eyebrows import change_eyebrow_style    # פונקציה שמשנה צבע שפתיים

# ====== SETTINGS ======
INPUT_FOLDER = r"C:\Users\halev\OneDrive\university\engineering_project\Engineering-Project\input_faces"
OUTPUT_FOLDER = r"C:\Users\halev\OneDrive\university\engineering_project\Engineering-Project\output_faces_change_eyebrows"
MASK_FOLDER = r"C:\Users\halev\OneDrive\university\engineering_project\Engineering-Project\eyebrow_masks"
NEW_EYEBROW_STYLE = "thick black eyebrows"
# ======================

# צור תיקיות אם לא קיימות
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)



def process_all_images():
    print("\n🚀 Starting batch lip color pipeline...\n")

    images = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not images:
        print("No images found in input folder.")
        return

    for img_name in images:
        try:
            print(f"\n🖼 Processing: {img_name}")

            input_path = os.path.join(INPUT_FOLDER, img_name)

            # ===== 1. CREATE MASK =====
            mask_output_path = os.path.join(MASK_FOLDER, img_name.split('.')[0] + "_eyebrows_mask.png")
            print("Creating eyebrows mask...")
            create_eyebrows_mask(input_path, mask_output_path)

            # ===== 2. CHANGE LIP COLOR =====
            edited_output_path = os.path.join(OUTPUT_FOLDER, img_name.split('.')[0] + "_eyebrows_edited.png")
            print("Changing eyebrows style...")
            change_eyebrow_style(input_path, mask_output_path, NEW_EYEBROW_STYLE, edited_output_path)

            print(f"✅ Done: {img_name}")

        except Exception as e:
            print(f"❌ Failed on {img_name}: {e}")


if __name__ == "__main__":
    process_all_images()
