import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv('Anemia.csv')

# Split the dataset into features (X) and labels (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE for over-sampling
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize the classifier
clf = RandomForestClassifier(random_state=42)

# Define hyperparameters and their values for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Create a GridSearchCV object for hyperparameter tuning
grid_search = GridSearchCV(clf, param_grid, cv=10, n_jobs=-1)

# Fit the classifier with hyperparameter tuning using 10-fold cross-validation
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best estimator after hyperparameter tuning
best_classifier = grid_search.best_estimator_

# Perform 10-fold cross-validation and calculate the mean accuracy
cv_scores = cross_val_score(best_classifier, X_train_resampled, y_train_resampled, cv=10)
mean_cv_accuracy = np.mean(cv_scores)

# Predict labels for the testing data
y_pred = best_classifier.predict(X_test)

# Calculate the accuracy of the model on the test data
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Test Data:", accuracy)

# Print the classification report
report = classification_report(y_test, y_pred, zero_division=1)
print("Classification Report:\n", report)

# Print the mean cross-validation accuracy
print("Mean Cross-Validation Accuracy:", mean_cv_accuracy)
