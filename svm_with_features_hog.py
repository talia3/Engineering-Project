import os
import cv2
import numpy as np
from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Settings
DATADIR = r"C:\Users\97258\engineering_try_2\Engineering-Project"
CATEGORIES = [
    "No_Manipulation",
    "output_faces_change_lip_color_no_padding",
    "output_faces_change_eye_color_no_padding"
]
IMG_SIZE = 128

# Extract HOG features from image
def get_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    
    return features


# Load data
def load_data():
    data = []
    labels = []
    
    print("Loading images and extracting HOG features...")
    
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        
        if not os.path.exists(path):
            print(f"Warning: Directory {path} not found.")
            continue
        
        count = 0
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                image = cv2.imread(img_path)
                
                if image is None:
                    continue
                
                # Resize image
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                
                # Extract HOG features
                features = get_hog_features(image)
                
                data.append(features)
                labels.append(class_num)
                count += 1
                
            except Exception as e:
                pass
        
        print(f"Loaded {count} images from {category}")
    
    return np.array(data), np.array(labels)


# Main execution
print("=" * 60)
print("SVM with HOG Features (No Padding)")
print("=" * 60)

X, y = load_data()

print(f"\nTotal samples: {len(X)}")
print(f"Feature vector size: {X.shape[1]}")

if len(X) > 0:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create and train SVM model
    model = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        verbose=1
    )
    
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Results
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "=" * 60)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("=" * 60)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CATEGORIES))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    import joblib

    joblib.dump(model, "svm_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("Model saved!")
else:
    print("No images found!")
