import os
from lips_mask import create_lips_mask  # פונקציה שיוצרת מסכה
from change_lips_color import change_lip_color    # פונקציה שמשנה צבע שפתיים

# ====== SETTINGS ======
INPUT_FOLDER = r"C:\Users\halev\OneDrive\university\engineering_project\Engineering-Project\input_faces"
OUTPUT_FOLDER = r"C:\Users\halev\OneDrive\university\engineering_project\Engineering-Project\output_faces_change_lip_color"
MASK_FOLDER = r"C:\Users\halev\OneDrive\university\engineering_project\Engineering-Project\lip_masks"
NEW_LIP_COLOR = "matte pink lipstick"
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
            mask_output_path = os.path.join(MASK_FOLDER, img_name.split('.')[0] + "_lips_mask.png")
            print("Creating lips mask...")
            create_lips_mask(input_path, mask_output_path)

            # ===== 2. CHANGE LIP COLOR =====
            edited_output_path = os.path.join(OUTPUT_FOLDER, img_name.split('.')[0] + "_lips_edited.png")
            print("Changing lip color...")
            change_lip_color(input_path, mask_output_path, NEW_LIP_COLOR, edited_output_path)

            print(f"✅ Done: {img_name}")

        except Exception as e:
            print(f"❌ Failed on {img_name}: {e}")


if __name__ == "__main__":
    process_all_images()
