
import os

# ==== הנתיב לתיקיית התמונות שלך ====
folder_path = r"C:\Users\97258\engineering_try_2\Engineering-Project\good_image"   # 🔁 שנה לכאן

# סיומות של קבצי תמונה
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

# קבלת רשימת קבצים שהם תמונות בלבד
images = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

# מיון לפי שם קובץ
images.sort()

print(f"Found {len(images)} images. Starting renaming...")

for index, filename in enumerate(images, start=1):
    old_path = os.path.join(folder_path, filename)

    # שומר על הסיומת המקורית
    ext = os.path.splitext(filename)[1]
    new_filename = f"{index}{ext}"
    new_path = os.path.join(folder_path, new_filename)

    os.rename(old_path, new_path)
    print(f"{filename}  -->  {new_filename}")

print("Done! 🎉")
