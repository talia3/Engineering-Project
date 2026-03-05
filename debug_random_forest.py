#!/usr/bin/env python3
import sys
print("Starting debug...", file=sys.stdout, flush=True)

try:
    print("Loading module...", file=sys.stdout, flush=True)
    import os
    import cv2
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    print("Modules loaded successfully", file=sys.stdout, flush=True)
    
    # --- הגדרות נתיבים ופרמטרים ---
    DATADIR = r"C:\Users\97258\engineering_try_2\Engineering-Project"
    CATEGORIES = ["No_Manipulation", "output_faces_change_lip_color", "output_faces_change_eye_color"]
    IMG_SIZE = 64
    
    def load_data():
        data = []
        labels = []
        
        print("טוען תמונות...", file=sys.stdout, flush=True)
        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)
            class_num = CATEGORIES.index(category)
            
            print(f"Checking path: {path}", file=sys.stdout, flush=True)
            
            if not os.path.exists(path):
                print(f"אזהרה: התיקייה {path} לא נמצאה.", file=sys.stdout, flush=True)
                continue
            
            print(f"Loading images from {category}...", file=sys.stdout, flush=True)
                
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img))
                    if img_array is None:
                        continue
                    
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    data.append(new_array.flatten())
                    labels.append(class_num)
                except Exception as e:
                    print(f"שגיאה בטעינת תמונה {img}: {e}", file=sys.stdout, flush=True)
            
            print(f"Loaded {len([l for l in labels if l == class_num])} images from {category}", file=sys.stdout, flush=True)
                
        return np.array(data), np.array(labels)
    
    print("About to load data...", file=sys.stdout, flush=True)
    X, y = load_data()
    
    print(f"Data loaded. Shape: {X.shape}, Labels: {y.shape}", file=sys.stdout, flush=True)
    
    if len(X) == 0:
        print("לא נמצאו תמונות. בדקי את נתיב התיקייה.", file=sys.stdout, flush=True)
    else:
        print("Starting training...", file=sys.stdout, flush=True)
        X = X / 255.0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"מתחיל אימון על {len(X_train)} תמונות...", file=sys.stdout, flush=True)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
        rf_model.fit(X_train, y_train)
        
        print("Training complete. Making predictions...", file=sys.stdout, flush=True)
        y_pred = rf_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n" + "="*30, file=sys.stdout, flush=True)
        print(f"תוצאות המודל:", file=sys.stdout, flush=True)
        print(f"אחוז דיוק כללי: {accuracy * 100:.2f}%", file=sys.stdout, flush=True)
        print("="*30, file=sys.stdout, flush=True)
        
        print("\nדוח סיווג מפורט:", file=sys.stdout, flush=True)
        print(classification_report(y_test, y_pred, target_names=CATEGORIES), file=sys.stdout, flush=True)
        
        print("\nמטריצת בלבול (Confusion Matrix):", file=sys.stdout, flush=True)
        print(confusion_matrix(y_test, y_pred), file=sys.stdout, flush=True)

except Exception as e:
    print(f"Error occurred: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc()
