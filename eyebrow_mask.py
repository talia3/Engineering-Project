import cv2
import numpy as np
import os

def create_eyebrows_mask(image_path, output_path):
    print(f"Starting eyebrow detection for: {os.path.basename(image_path)}")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return False

    height, width = img.shape[:2]
    # Create white mask (255), we will draw black (0) for the eyebrows
    mask = np.ones((height, width), dtype=np.uint8) * 255
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- METHOD 1: MediaPipe (High Accuracy) ---
    try:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        model_path = 'face_landmarker.task' 
        if os.path.exists(model_path):
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
            
            with vision.FaceLandmarker.create_from_options(options) as landmarker:
                mp_image = python.Image(image_format=python.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                result = landmarker.detect(mp_image)
                
                if result.face_landmarks:
                    print("Using MediaPipe landmarks...")
                    # Left Eyebrow indices (inner to outer)
                    LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
                    # Right Eyebrow indices (inner to outer)
                    RIGHT_EYEBROW = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
                    
                    landmarks = result.face_landmarks[0]
                    
                    for brow_indices in [LEFT_EYEBROW, RIGHT_EYEBROW]:
                        points = []
                        for idx in brow_indices:
                            pt = landmarks[idx]
                            points.append([int(pt.x * width), int(pt.y * height)])
                        
                        cv2.fillPoly(mask, [np.array(points)], 0)
                    
                    cv2.imwrite(output_path, mask)
                    print(f"Success via MediaPipe: {output_path}")

                    visualization = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    output_viz = output_path.replace('.jpg', '_visualization.jpg').replace('.png', '_visualization.png')
                    combined = cv2.addWeighted(img, 0.7, visualization, 0.3, 0)
                    cv2.imwrite(output_viz, combined)
                    print(f"Visualization saved to: {output_viz}")

                    return True
    except Exception as e:
        print(f"MediaPipe Method skipped: {e}")

# --- METHOD 2: Haar Cascade (Corrected Alignment) ---
    print("Adjusting Haar ROI for better brow coverage...")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 10)
    
    if len(eyes) > 0:
        for (ex, ey, ew, eh) in eyes:
            # Width: 1.5x eye width to catch the outer 'tail' of the brow
            brow_w = int(ew * 1.5)
            # Height: Increase to 80% of eye height to ensure full coverage
            brow_h = int(eh * 0.8)
            # X: Shift left by 25% of eye width to center the wider box
            brow_x = ex - int(ew * 0.25)
            # Y: Shift DOWN from your previous version. 
            # Moving up by only 40% of eye height should land right on the brow.
            brow_y = ey - int(eh * 0.4) 
            
            # Draw the box
            cv2.rectangle(mask, (brow_x, brow_y), (brow_x + brow_w, brow_y + brow_h), 0, -1)
        
        # Add a slight blur to the mask edges to help the AI blend the hair with skin
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        cv2.imwrite(output_path, mask)
        print(f"Success via Aggressive Haar: {output_path}")

        visualization = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        output_viz = output_path.replace('.jpg', '_visualization.jpg').replace('.png', '_visualization.png')
        combined = cv2.addWeighted(img, 0.7, visualization, 0.3, 0)
        cv2.imwrite(output_viz, combined)
        print(f"Visualization saved to: {output_viz}")
        return True
    
    print("Failed to detect eyebrows via any method.")
    return False

if __name__ == "__main__":
    # Update these paths for your environment
    input_img = r"C:\Users\halev\OneDrive\university\engineering_project\Engineering-Project\tryIn\000002.jpg"
    output_dir = r"C:\Users\halev\OneDrive\university\engineering_project\Engineering-Project\tryOut"
    
    # Ensure filename is handled
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = os.path.splitext(os.path.basename(input_img))[0]
    final_output = os.path.join(output_dir, f"{filename}_eyebrows_mask.jpg")
    
    create_eyebrows_mask(input_img, final_output)