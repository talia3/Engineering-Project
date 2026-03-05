import os
import cv2
import numpy as np

from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------
# Settings
# -----------------------------
DATADIR = r"C:\Users\97258\engineering_try_2\Engineering-Project"

CATEGORIES = [
    "No_Manipulation",
    "output_faces_change_lip_color_no_padding",
    "output_faces_change_eye_color_no_padding"
]

IMG_SIZE = 160


# -----------------------------
# HOG feature extractor
# -----------------------------
def extract_features(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm='L2-Hys'
    )

    return hog_features


# -----------------------------
# Load dataset
# -----------------------------
def load_data():

    data = []
    labels = []

    print("Loading images...")

    for category in CATEGORIES:

        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)

        count = 0

        for img_name in os.listdir(path):

            try:
                img_path = os.path.join(path, img_name)

                image = cv2.imread(img_path)

                if image is None:
                    continue

                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

                features = extract_features(image)

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
print("KNN Image Classification")
print("="*50)

X, y = load_data()

print(f"\nTotal samples: {len(X)}")
print(f"Feature size: {X.shape[1]}")


# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")


# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -----------------------------
# Find best K using GridSearch
# -----------------------------
param_grid = {
    'n_neighbors': [3,5,7,9,11,15],
    'weights': ['uniform','distance'],
    'metric': ['euclidean','manhattan']
}

knn = KNeighborsClassifier()

grid = GridSearchCV(
    knn,
    param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

print("\nSearching best parameters...")
grid.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid.best_params_)

model = grid.best_estimator_


# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n"+"="*50)
print(f"Accuracy: {accuracy*100:.2f}%")
print("="*50)


print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=CATEGORIES))


print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))