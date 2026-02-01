import os
from final_create_mask_on_eye import create_mask  # פונקציה שיוצרת מסכה
from change_eye_color import change_eye_color    # פונקציה שמשנה צבע עיניים

# ====== SETTINGS ======
INPUT_FOLDER = r"C:\Users\halev\OneDrive\university\engineering_project\Engineering-Project\input_faces"
OUTPUT_FOLDER = r"C:\Users\halev\OneDrive\university\engineering_project\Engineering-Project\output_faces_change_eye_color"
MASK_FOLDER = r"C:\Users\halev\OneDrive\university\engineering_project\Engineering-Project\eye_masks"
#1-12:
#NEW_EYE_COLOR = "bright natural blue eyes"
#13-20:
#NEW_EYE_COLOR = "bright natural blue or green eyes"
#21-28:
#NEW_EYE_COLOR = "bright natural blue or bright natural green eyes, keep it realistic"
#29-34:
#NEW_EYE_COLOR = "Change the iris color to a soft, light natural blue or green. Keep the original eye texture, gradients, shadows, and reflections. The color should be slightly desaturated for realism, not vivid, not glowing, and not oversaturated. The result must look like a naturally occurring human eye color. Apply the change only inside the eye mask and do not alter skin, lighting, or any other facial features."
#35-43:
NEW_EYE_COLOR = "Change the iris color to a light natural green or blue with realistic human pigmentation. Preserve iris fibers, radial patterns, limbal ring, shadows, and natural light reflections. Blend the color with the original eye texture instead of flat recoloring. Slight desaturation for realism. No glow, no neon tones, no solid fill, no fantasy effect. The eye should keep depth and natural variation in color intensity. Apply the edit only inside the eye mask. Do not modify skin, sclera, lighting, or facial features.Reduce color strength to match real-world eye pigmentation levels."
# ======================

# צור תיקיות אם לא קיימות
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)



def process_all_images():
    print("\n🚀 Starting batch eye color pipeline...\n")

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
            print("Creating eye mask...")
            create_mask(input_path, mask_output_path)

            # ===== 2. CHANGE EYE COLOR =====
            edited_output_path = os.path.join(OUTPUT_FOLDER, img_name.split('.')[0] + "_edited.png")
            print("Changing eye color...")
            change_eye_color(input_path, mask_output_path, NEW_EYE_COLOR, edited_output_path)

            print(f"✅ Done: {img_name}")

        except Exception as e:
            print(f"❌ Failed on {img_name}: {e}")


if __name__ == "__main__":
    process_all_images()
