import os
import cv2
import numpy as np
from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Import mediapipe for face detection
import mediapipe as mp

# Initialize MediaPipe Face Mesh
if not hasattr(mp, 'solutions'):
    # If the standard import doesn't work, use alternative
    from mediapipe.python.solutions import face_mesh
    mp_face_mesh = face_mesh
else:
    mp_face_mesh = mp.solutions.face_mesh


# -----------------------------
# Settings
# -----------------------------
DATADIR = r"C:\Users\97258\engineering_try_2\Engineering-Project"

CATEGORIES = [
    "No_Manipulation",
    "output_faces_change_lip_color_no_padding",
    "output_faces_change_eye_color_no_padding"
]

IMG_SIZE = 128


# -----------------------------
# MediaPipe Face Mesh
# -----------------------------
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)


# Face region indices
LEFT_EYE = [33,133,160,159,158,157,173]
RIGHT_EYE = [362,263,387,386,385,384,398]
MOUTH = [61,146,91,181,84,17,314,405,321]


# -----------------------------
# Function to extract a region from the face
# -----------------------------
def extract_region(image, landmarks, indices):

    h, w, _ = image.shape

    pts = []

    for i in indices:
        lm = landmarks[i]
        pts.append((int(lm.x * w), int(lm.y * h)))

    pts = np.array(pts)

    x, y, w_box, h_box = cv2.boundingRect(pts)

    region = image[y:y+h_box, x:x+w_box]

    if region.size == 0:
        return None

    region = cv2.resize(region,(64,64))

    return region


# -----------------------------
# HOG features
# -----------------------------
def get_hog_features(image):

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm='L2-Hys'
    )

    return features


# -----------------------------
# טעינת הדאטה
# -----------------------------
def load_data():

    data = []
    labels = []

    print("Loading images...")

    for category in CATEGORIES:

        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)

        if not os.path.exists(path):
            continue

        for img_name in os.listdir(path):

            try:

                img_path = os.path.join(path,img_name)

                image = cv2.imread(img_path)

                if image is None:
                    continue

                image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))

                rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

                results = face_mesh.process(rgb)

                if not results.multi_face_landmarks:
                    continue

                landmarks = results.multi_face_landmarks[0].landmark

                # Extract regions
                left_eye = extract_region(image,landmarks,LEFT_EYE)
                right_eye = extract_region(image,landmarks,RIGHT_EYE)
                mouth = extract_region(image,landmarks,MOUTH)

                if left_eye is None or right_eye is None or mouth is None:
                    continue

                # Features
                f1 = get_hog_features(left_eye)
                f2 = get_hog_features(right_eye)
                f3 = get_hog_features(mouth)

                features = np.concatenate([f1,f2,f3])

                data.append(features)
                labels.append(class_num)

            except:
                pass

    return np.array(data), np.array(labels)


# Load data - main execution
X,y = load_data()

print("Total samples:",len(X))

X_train,X_test,y_train,y_test = train_test_split(
    X,y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# SVM Model
model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale'
)

print("Training model...")
model.fit(X_train,y_train)


# Make predictions
y_pred = model.predict(X_test)


# Results
accuracy = accuracy_score(y_test,y_pred)

print("\nAccuracy:",accuracy)

print("\nClassification Report:\n")

print(classification_report(y_test,y_pred,target_names=CATEGORIES))