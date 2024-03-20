import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
import warnings

df = pd.read_csv('Anemia.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

clf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train_resampled, y_train_resampled)

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

clf = RandomForestClassifier(random_state=42, **best_params)

clf.fit(X_train_resampled, y_train_resampled)

true_labels = []
predicted_labels = []

cap = cv2.VideoCapture(0)

num_images_captured = 0
num_images_to_capture = 100

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in eyes:
        eye_image = frame[y:y + h, x:x + w]

        resized_eye = cv2.resize(eye_image, (200, 200))

        if key == ord('s') and num_images_captured < num_images_to_capture:
            eye_feature_vector = []

            for channel in cv2.split(resized_eye):
                mean_value = np.mean(channel)
                eye_feature_vector.append(mean_value)

            label = clf.predict([eye_feature_vector])[0]

            predicted_labels.append(label)
            true_labels.append(1)  

            num_images_captured += 1

            if num_images_captured >= num_images_to_capture:
                print("Collected 100 images")
                break

            print(label)
            cv2.putText(frame, str(label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Eye Detection', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()

