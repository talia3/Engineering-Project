import cv2
import numpy as np
import os
import math


def create_pupil_mask(image_path, output_path):
    print(f"Starting pupil mask detection for: {os.path.basename(image_path)}")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return False

    height, width = img.shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- METHOD 1: MediaPipe (Use Iris Landmarks to get full pupil) ---
    try:
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
        from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
        
        model_path = 'face_landmarker.task'
        if os.path.exists(model_path):
            base_options = BaseOptions(model_asset_path=model_path)
            options = FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=RunningMode.IMAGE,
                num_faces=1
            )
            
            with FaceLandmarker.create_from_options(options) as landmarker:
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_image)
                result = landmarker.detect(mp_image)
                
                if result.face_landmarks:
                    print("Using MediaPipe iris landmarks for full pupil...")
                    landmarks = result.face_landmarks[0]

                    # Iris indices
                    LEFT_IRIS  = [468, 469, 470, 471, 472]
                    RIGHT_IRIS = [473, 474, 475, 476, 477]

                    for iris in [LEFT_IRIS, RIGHT_IRIS]:
                        points = np.array([[int(landmarks[idx].x * width),
                                            int(landmarks[idx].y * height)] for idx in iris])
                        
                        # חשב מרכז ורדיוס עיגול
                        cx = int(np.mean(points[:,0]))
                        cy = int(np.mean(points[:,1]))
                        # מרחק מקסימלי מנקודת הקשתית למרכז
                        radius = int(max([math.hypot(cx - px, cy - py) for px, py in points]))
                        # צייר עיגול שמכסה את כל האישון
                        cv2.circle(mask, (cx, cy), radius, 0, -1)

                    cv2.imwrite(output_path, mask)
                    print(f"✅ Success via MediaPipe: {output_path}")
                    return True
    except Exception as e:
        print(f"MediaPipe Method skipped: {e}")

    # --- METHOD 2: Haar Cascade (Fallback) ---
    print("Trying Haar Cascade fallback...")
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex, ey, ew, eh) in eyes:
            center = (x + ex + ew//2, y + ey + eh//2)
            # רדיוס גדול לכיסוי כל האישון
            radius = int(min(ew, eh) * 0.45)
            cv2.circle(mask, center, radius, 0, -1)

    cv2.imwrite(output_path, mask)
    print(f"✅ Success via Haar Estimation: {output_path}")
    return True


if __name__ == "__main__":
    print("=== Pupil Mask Generator ===")
    
    input_path = r"tryIn\000003.jpg"
    output_path = r"tryOut"
    
    if os.path.isdir(output_path):
        input_filename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_path, f"{input_filename}_pupil_mask.jpg")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    create_pupil_mask(input_path, output_path)
    print("\nDone!")
