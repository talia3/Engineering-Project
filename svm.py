import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# --- Settings ---
DATADIR = r"C:\Users\97258\engineering_try_2\Engineering-Project"
CATEGORIES = ["No_Manipulation", "output_faces_change_lip_color_no_padding", "output_faces_change_eye_color_no_padding"]
IMG_SIZE = 64 

def load_data():
    data = []
    labels = []
    print("Loading data for SVM...")
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        if not os.path.exists(path): continue
            
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                if img_array is None: continue
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                # Flatten image to a single long array
                data.append(new_array.flatten())
                labels.append(class_num)
            except Exception:
                pass
    return np.array(data), np.array(labels)

# 1. Data preparation
X, y = load_data()
if len(X) > 0:
    X = X / 255.0  # Normalization - critical for SVM!
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    pca = PCA(n_components=150)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    # 2. Create SVM model
    # kernel='linear' is usually a good start for images
    # C=1.0 is the Regularization parameter
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)

    # 3. Train the model
    print("Starting SVM training (this may take a while)...")
    svm_model.fit(X_train, y_train)

    # 4. Prediction and evaluation
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "="*30)
    print(f"SVM Model Results:")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print("="*30)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CATEGORIES))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))