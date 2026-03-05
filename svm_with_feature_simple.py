import os
import cv2
import numpy as np
from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Settings
# -----------------------------
DATADIR = r"C:\Users\97258\engineering_try_2\Engineering-Project"
CATEGORIES = [
    "No_Manipulation",
    "output_faces_change_lip_color_no_padding",
    "output_faces_change_eye_color_no_padding"
]
IMG_SIZE = 160  # Resize all images

# -----------------------------
# Feature extraction
# -----------------------------
def get_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm='L2-Hys'
    )
    return features

def get_color_histogram(image):
    # Calculate 32-bin histogram for each channel
    hist_b = cv2.calcHist([image],[0],None,[32],[0,256])
    hist_g = cv2.calcHist([image],[1],None,[32],[0,256])
    hist_r = cv2.calcHist([image],[2],None,[32],[0,256])
    hist = np.concatenate([hist_b,hist_g,hist_r]).flatten()
    hist = hist / np.sum(hist)  # Normalize
    return hist

# -----------------------------
# Load data
# -----------------------------
def load_data():
    data = []
    labels = []

    print("Loading images and extracting features...\n")

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        if not os.path.exists(path):
            print(f"Directory not found: {path}")
            continue

        count = 0
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path,img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))

                # Extract HOG + Color features
                hog_features = get_hog_features(image)
                color_features = get_color_histogram(image)
                features = np.concatenate([hog_features,color_features])

                data.append(features)
                labels.append(class_num)
                count += 1
            except:
                pass
        print(f"{category}: {count} images loaded")

    return np.array(data), np.array(labels)

# -----------------------------
# Main
# -----------------------------
print("="*50)
print("SVM with HOG + Color Histogram + PCA")
print("="*50)

X, y = load_data()

print("\nTotal samples:",len(X))
print("Feature vector size:",X.shape[1])

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
print("\nTraining samples:",len(X_train))
print("Test samples:",len(X_test))

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA - reduce to 95% variance or max available components
print("\nApplying PCA...")
n_components = min(100, X_train.shape[0] - 1)  # Max components based on training samples
pca = PCA(n_components=n_components)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print("New feature size after PCA:",X_train.shape[1])
print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

# SVM model
model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    probability=True,
    verbose=True
)

# Train
print("\nTraining SVM model...\n")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\n"+"="*50)
print(f"Accuracy: {accuracy*100:.2f}%")
print("="*50)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=CATEGORIES))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))