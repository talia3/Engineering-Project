import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Path and Parameter Definitions ---
# Replace the path with the location of your directories
DATADIR = r"C:\Users\97258\engineering_try_2\Engineering-Project"
CATEGORIES = ["No_Manipulation", "output_faces_change_lip_color_no_padding", "output_faces_change_eye_color_no_padding"]
IMG_SIZE = 64  # Image size (64x64 pixels)

def load_data():
    data = []
    labels = []
    
    print("Loading images...")
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        
        # Check if the path exists
        if not os.path.exists(path):
            print(f"Warning: Directory {path} not found.")
            continue
            
        for img in os.listdir(path):
            try:
                # Reading image in color (RGB)
                img_array = cv2.imread(os.path.join(path, img))
                if img_array is None:
                    continue
                
                # Resize image to uniform size
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                
                # Convert image to flat array (Flatten)
                data.append(new_array.flatten())
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading image {img}: {e}")
                
    return np.array(data), np.array(labels)

# 1. Load data
X, y = load_data()

if len(X) == 0:
    print("No images found. Check the directory path.")
else:
    # 2. Normalize data (values between 0 and 1)
    X = X / 255.0

    # 3. Split into training (80%) and test (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Create and train Random Forest model
    print(f"Starting training with {len(X_train)} images...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # 5. Make predictions on test data
    y_pred = rf_model.predict(X_test)

    # 6. Print results
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*30)
    print(f"Model Results:")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print("="*30)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CATEGORIES, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the trained model
    joblib.dump(rf_model, "random_forest_model.pkl")
    print("\n✓ Model saved as 'random_forest_model.pkl'")