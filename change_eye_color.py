from openai import OpenAI
import requests
from PIL import Image
import os
import time


# ============= הגדרות - ערוך כאן =============
API_KEY = r"sk-xxxx"  
IMAGE_PATH = r"C:\Users\97258\project1\Engineering-Project\tryIn\000002.jpg"  # נתיב לתמונת הפנים
MASK_PATH = r"C:\Users\97258\project1\Engineering-Project\tryOut\good_try\000002_mask.jpg"  # נתיב לתמונת המסכה (אישונים בשחור, שאר הפנים בלבן)
NEW_COLOR = "bright green eyes"  # תיאור הצבע החדש של העיניים
OUTPUT_PATH = r"C:\Users\97258\project1\Engineering-Project\tryOut\good_try\edited_image2.png"  # נתיב לשמירת התמונה הערוכה
# ============================================



def prepare_mask(mask_path):
    """
    Black in mask → transparent → area to edit
    White in mask → opaque → keep original
    """
    print(f"Preparing edit mask from: {mask_path}")

    with Image.open(mask_path).convert("L") as mask:
        w, h = mask.size
        new_mask = Image.new("RGBA", (w, h), (255, 255, 255, 255))

        for y in range(h):
            for x in range(w):
                value = mask.getpixel((x, y))
                if value < 100:  # dark = edit region
                    new_mask.putpixel((x, y), (255, 255, 255, 0))
                else:
                    new_mask.putpixel((x, y), (255, 255, 255, 255))

        new_path = os.path.splitext(mask_path)[0] + "_editmask.png"
        new_mask.save(new_path, "PNG")

    print(f"Mask ready: {new_path}")
    return new_path


def convert_image_to_rgba(image_path):
    """Convert main image to RGBA PNG"""
    print(f"Converting image to RGBA: {image_path}")

    with Image.open(image_path) as img:
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        new_path = os.path.splitext(image_path)[0] + "_temp.png"
        img.save(new_path, "PNG")

    return new_path


def safe_remove(path):
    try:
        time.sleep(0.5)
        if os.path.exists(path):
            os.remove(path)
    except:
        pass


def change_eye_color(image_path, mask_path, new_color, output_path):
    client = OpenAI(api_key=API_KEY)

    image_temp = None
    mask_temp = None

    try:
        image_temp = convert_image_to_rgba(image_path)
        mask_temp = prepare_mask(mask_path)  # 🔥 המסכה הנכונה

        print("\nSending request to OpenAI...")

        with open(image_temp, "rb") as img_file, open(mask_temp, "rb") as mask_file:
            response = client.images.edit(
                model="dall-e-2",
                image=img_file,
                mask=mask_file,
                prompt=f"Photorealistic edit. Change ONLY the iris color to {new_color}. Keep face, skin, lighting and photo identical.",
                size="1024x1024"
            )

        image_url = response.data[0].url
        print("Edit success. Downloading...")

        img_data = requests.get(image_url, timeout=30).content
        with open(output_path, "wb") as f:
            f.write(img_data)

        print(f"\n✅ Saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

    finally:
        safe_remove(image_temp)
        safe_remove(mask_temp)


# ================= RUN =================
if __name__ == "__main__":
    print("Starting eye color edit...\n")

    result = change_eye_color(IMAGE_PATH, MASK_PATH, NEW_COLOR, OUTPUT_PATH)

    if result:
        print("🎉 Done!")
    else:
        print("✗ Failed.")
