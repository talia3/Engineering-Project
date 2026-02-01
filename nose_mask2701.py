import cv2
import numpy as np
import os

def create_nose_mask_geometric(image_path, output_path='nose_mask.jpg'):
    """
    Creates nose mask using improved geometric estimation based on eyes.
    More accurate than basic method.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return False
    
    height, width = image.shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        print("No face detected")
        return False
    
    for (fx, fy, fw, fh) in faces:
        # Detect eyes
        roi_gray = gray[fy:fy+fh, fx:fx+fw]
        upper_face_height = int(fh * 0.6)
        roi_gray_upper = roi_gray[0:upper_face_height, :]
        
        eyes = eye_cascade.detectMultiScale(roi_gray_upper, 1.1, 3, minSize=(30, 30))
        
        nose_points = []
        
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate
            eyes_sorted = sorted(eyes, key=lambda e: e[0])
            left_eye = eyes_sorted[0]
            right_eye = eyes_sorted[-1]
            
            # Calculate eye centers
            left_x = fx + left_eye[0] + left_eye[2] // 2
            left_y = fy + left_eye[1] + left_eye[3] // 2
            right_x = fx + right_eye[0] + right_eye[2] // 2
            right_y = fy + right_eye[1] + right_eye[3] // 2
            
            # Nose geometry based on eyes
            eye_distance = right_x - left_x
            eye_center_y = (left_y + right_y) // 2
            nose_center_x = (left_x + right_x) // 2
            
            # Create nose shape with multiple points - ENLARGED
            # Nose bridge (top) - starts higher, between the eyes
            bridge_top_y = eye_center_y + int(eye_distance * 0.05)  # Much higher
            nose_points.append([nose_center_x, bridge_top_y])
            
            # Nose sides at bridge level - wider
            bridge_width = int(eye_distance * 0.2)  # Increased from 0.15
            nose_points.append([nose_center_x - bridge_width, bridge_top_y])
            nose_points.append([nose_center_x + bridge_width, bridge_top_y])
            
            # Upper middle points
            upper_middle_y = bridge_top_y + int(eye_distance * 0.25)
            upper_middle_width = int(eye_distance * 0.22)
            nose_points.append([nose_center_x - upper_middle_width, upper_middle_y])
            nose_points.append([nose_center_x + upper_middle_width, upper_middle_y])
            
            # Middle points
            middle_y = bridge_top_y + int(eye_distance * 0.45)
            middle_width = int(eye_distance * 0.24)
            nose_points.append([nose_center_x - middle_width, middle_y])
            nose_points.append([nose_center_x + middle_width, middle_y])
            
            # Lower middle points
            lower_middle_y = bridge_top_y + int(eye_distance * 0.6)
            lower_middle_width = int(eye_distance * 0.26)
            nose_points.append([nose_center_x - lower_middle_width, lower_middle_y])
            nose_points.append([nose_center_x + lower_middle_width, lower_middle_y])
            
            # Nose tip area
            tip_y = bridge_top_y + int(eye_distance * 0.75)
            tip_width = int(eye_distance * 0.25)
            nose_points.append([nose_center_x, tip_y])
            nose_points.append([nose_center_x - tip_width, tip_y])
            nose_points.append([nose_center_x + tip_width, tip_y])
            
            # Nose base (nostrils area) - wider and lower
            base_y = tip_y + int(eye_distance * 0.2)
            base_width = int(eye_distance * 0.28)
            nose_points.append([nose_center_x - base_width, base_y])
            nose_points.append([nose_center_x + base_width, base_y])
            nose_points.append([nose_center_x, base_y])
            
            # Extra wide points at base for full coverage
            extra_base_y = base_y - int(eye_distance * 0.05)
            extra_base_width = int(eye_distance * 0.3)
            nose_points.append([nose_center_x - extra_base_width, extra_base_y])
            nose_points.append([nose_center_x + extra_base_width, extra_base_y])
            
            # Additional side points for complete coverage
            for i in range(3, 8):
                side_y = bridge_top_y + int(eye_distance * (0.1 * i))
                side_width = int(eye_distance * (0.18 + 0.02 * i))
                nose_points.append([nose_center_x - side_width, side_y])
                nose_points.append([nose_center_x + side_width, side_y])
            
        else:
            # Fallback if eyes not detected - ENLARGED
            nose_center_x = fx + fw // 2
            nose_top_y = fy + int(fh * 0.3)  # Higher start
            nose_height = int(fh * 0.4)  # Taller
            nose_width = int(fw * 0.2)  # Wider
            
            # Create larger nose shape with more points
            for i in range(10):
                progress = i / 9.0
                y = nose_top_y + int(nose_height * progress)
                # Width increases as we go down
                width = int(nose_width * (1.0 + progress * 0.8))
                nose_points.append([nose_center_x - width, y])
                nose_points.append([nose_center_x + width, y])
                nose_points.append([nose_center_x, y])
        
        nose_points = np.array(nose_points, dtype=np.int32)
        hull = cv2.convexHull(nose_points)
        cv2.fillPoly(mask, [hull], 0)
        
        print(f"Nose detected with geometric method ({len(nose_points)} points)")
    
    # Save results
    cv2.imwrite(output_path, mask)
    print(f"Nose mask saved to: {output_path}")
    
    visualization = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    output_viz = output_path.replace('.jpg', '_visualization.jpg').replace('.png', '_visualization.png')
    combined = cv2.addWeighted(image, 0.7, visualization, 0.3, 0)
    cv2.imwrite(output_viz, combined)
    print(f"Visualization saved to: {output_viz}")
    
    return True


def create_nose_mask_with_dlib(image_path, output_path='nose_mask.jpg'):
    """
    Creates accurate nose mask using dlib facial landmarks.
    Dlib provides 68 facial landmarks including precise nose points.
    """
    try:
        import dlib
    except ImportError:
        print("Error: dlib is not installed.")
        print("Install it using: pip install dlib")
        print("Note: dlib installation can be complex on Windows.")
        print("Alternative: pip install cmake, then pip install dlib")
        return False
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return False
    
    height, width = image.shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize dlib's face detector and facial landmarks predictor
    detector = dlib.get_frontal_face_detector()
    
    # Download shape predictor if needed
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        print("Downloading facial landmarks predictor...")
        print("Please download 'shape_predictor_68_face_landmarks.dat' from:")
        print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Extract it to the same directory as this script.")
        return False
    
    predictor = dlib.shape_predictor(predictor_path)
    
    # Detect faces
    faces = detector(gray)
    
    if len(faces) == 0:
        print("No face detected")
        return False
    
    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)
        
        # Nose landmarks in dlib's 68-point model:
        # Points 27-35 are the nose
        # 27-30: nose bridge
        # 31-35: nose base
        nose_points = []
        for i in range(27, 36):  # All nose points
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            nose_points.append([x, y])
        
        nose_points = np.array(nose_points, dtype=np.int32)
        
        # Create a filled polygon of the nose
        cv2.fillPoly(mask, [nose_points], 0)
        
        print(f"Nose detected with {len(nose_points)} landmark points")
    
    # Save results
    cv2.imwrite(output_path, mask)
    print(f"Nose mask saved to: {output_path}")
    
    visualization = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    output_viz = output_path.replace('.jpg', '_visualization.jpg').replace('.png', '_visualization.png')
    combined = cv2.addWeighted(image, 0.7, visualization, 0.3, 0)
    cv2.imwrite(output_viz, combined)
    print(f"Visualization saved to: {output_viz}")
    
    return True


def create_nose_mask_with_mediapipe(image_path, output_path='nose_mask.jpg'):
    """
    Creates accurate nose mask using MediaPipe Face Mesh.
    MediaPipe provides 478 facial landmarks including very precise nose points.
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError:
        print("Error: MediaPipe is not installed correctly.")
        return False
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return False
    
    height, width = image.shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Try with the old API first (if available)
    try:
        mp_face_mesh = mp.solutions.face_mesh
        
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        ) as face_mesh:
            
            results = face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                print("No face detected")
                return False
            
            for face_landmarks in results.multi_face_landmarks:
                # MediaPipe nose landmark indices (478-point model)
                nose_indices = [
                    1,    # Nose tip
                    2,    # Bottom center
                    98,   # Left nostril
                    327,  # Right nostril
                    168,  # Top of nose bridge
                    6,    # Middle of nose bridge
                    197, 195, 5,    # Left side of nose
                    419, 248, 456,  # Right side of nose
                    129, 203,       # Additional left points
                    358, 423,       # Additional right points
                    49, 279,        # Nose base left/right
                    64, 294,        # Additional base points
                ]
                
                nose_points = []
                for idx in nose_indices:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    nose_points.append([x, y])
                
                nose_points = np.array(nose_points, dtype=np.int32)
                
                # Calculate convex hull to get smooth nose shape
                hull = cv2.convexHull(nose_points)
                
                # Fill the nose area
                cv2.fillPoly(mask, [hull], 0)
                
                print(f"Nose detected with {len(nose_indices)} landmark points")
        
        # Save results
        cv2.imwrite(output_path, mask)
        print(f"Nose mask saved to: {output_path}")
        
        visualization = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        output_viz = output_path.replace('.jpg', '_visualization.jpg').replace('.png', '_visualization.png')
        combined = cv2.addWeighted(image, 0.7, visualization, 0.3, 0)
        cv2.imwrite(output_viz, combined)
        print(f"Visualization saved to: {output_viz}")
        
        return True
        
    except AttributeError:
        # Old API not available, use simplified geometric method
        print("MediaPipe old API not available, using geometric estimation with face detection...")
        return create_nose_mask_geometric(image_path, output_path)


def create_nose_mask(image_path, output_path='nose_mask.jpg'):
    """
    Main function that tries different methods for nose detection.
    """
    print("Attempting nose detection with facial landmarks...")
    
    # Try MediaPipe first (easier to install and very accurate)
    print("\n1. Trying MediaPipe Face Mesh...")
    success = create_nose_mask_with_mediapipe(image_path, output_path)
    
    if success:
        return True
    
    # Try dlib as fallback
    print("\n2. Trying dlib facial landmarks...")
    success = create_nose_mask_with_dlib(image_path, output_path)
    
    if success:
        return True
    
    print("\nFailed to detect nose with facial landmarks.")
    print("Please ensure you have MediaPipe or dlib installed.")
    return False


if __name__ == "__main__":
    print("=== Accurate Nose Mask Generator (Facial Landmarks) ===")
    print()
    
    # CONFIGURE YOUR PATHS HERE
    input_path = r"C:\Users\97258\project111Try\Engineering-Project\input_faces\DONE\WhatsApp Image 2026-01-28 at 11.18.38.jpeg"
    output_path = r"C:\Users\97258\project1\Engineering-Project\tryOut"
    
    # ====================================
    
    print(f"Input image: {input_path}")
    print(f"Output location: {output_path}")
    print()
    
    # Check if file exists
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        exit(1)
    
    # If output_path is a directory, create a default filename
    if os.path.isdir(output_path):
        input_filename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_path, f"{input_filename}_nose_mask.jpg")
        print(f"Output directory detected. Saving to: {output_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    print()
    print("Processing...")
    
    success = create_nose_mask(input_path, output_path)
    
    if not success:
        print("\nTo use this script, you need MediaPipe:")
        print("  pip install mediapipe")
        print("\nMediaPipe provides 478 facial landmarks for very accurate detection!")
    
    print("\nDone!")