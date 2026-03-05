import os
import cv2
import numpy as np
from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================================
# SETTINGS
# =========================================

DATADIR = r"C:\Users\97258\engineering_try_2\Engineering-Project"

CATEGORIES = [
    "No_Manipulation",
    "output_faces_change_lip_color",
    "output_faces_change_eye_color"
]

IMG_SIZE = 160


# =========================================
# FEATURE EXTRACTION
# =========================================

def extract_features(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm='L2-Hys'
    )

    # Color histogram
    hist = cv2.calcHist([image],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
    hist = cv2.normalize(hist,hist).flatten()

    features = np.concatenate((hog_features, hist))

    return features


# =========================================
# LOAD DATA
# =========================================

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

                image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))

                features = extract_features(image)

                data.append(features)
                labels.append(class_num)

                count += 1

            except:
                pass

        print(f"{category}: {count} images")

    return np.array(data), np.array(labels)


# =========================================
# LOAD DATASET
# =========================================

X, y = load_data()

print("\nTotal samples:", len(X))
print("Feature size:", X.shape[1])


# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# =========================================
# TRAIN BINARY SVM MODELS
# =========================================

models = []

print("\nTraining Binary SVM models...")

for class_id in range(len(CATEGORIES)):

    print(f"\nTraining classifier for: {CATEGORIES[class_id]}")

    y_binary = (y_train == class_id).astype(int)

    model = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True
    )

    model.fit(X_train, y_binary)

    models.append(model)


# =========================================
# PREDICTION
# =========================================

print("\nPredicting...")

y_pred = []

for sample in X_test:

    probs = []

    for model in models:

        prob = model.predict_proba(sample.reshape(1,-1))[0][1]

        probs.append(prob)

    prediction = np.argmax(probs)

    y_pred.append(prediction)


y_pred = np.array(y_pred)


# =========================================
# RESULTS
# =========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n"+"="*50)
print(f"Accuracy: {accuracy*100:.2f}%")
print("="*50)

print("\nClassification Report:\n")

print(classification_report(
    y_test,
    y_pred,
    target_names=CATEGORIES
))

print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test,y_pred))