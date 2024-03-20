# Anemia-Detection-using-conjunctiva-images
AI-Based Anemia Detection Using Conjunctiva Images


This project focuses on developing an individualized AI solution for detecting anemia using images of the conjunctiva. Anemia, a condition characterized by a deficiency of red blood cells or hemoglobin, can have serious health consequences if left untreated. Early detection is critical for effective management and prevention of complications.

The project employs machine learning techniques, specifically utilizing the Random Forest classifier, to analyze conjunctiva images and predict the presence of anemia. The process involves several key steps:

1. Data Collection: A dataset of conjunctiva images is collected from individuals, including both anemic and non-anemic cases. The images are acquired using standard imaging equipment in a controlled environment to ensure consistency.

2. Preprocessing: Preprocessing techniques are applied to enhance the quality of the conjunctiva images and remove any noise. This may involve resizing, normalization, and contrast adjustment to standardize the images for analysis.

3. Feature Extraction: Relevant features are extracted from the conjunctiva images, such as color histograms, texture descriptors, and other image characteristics. These features serve as input variables for the machine learning model.

4. SMOTE (Synthetic Minority Over-sampling Technique): Since anemia cases may be underrepresented in the dataset, SMOTE is applied to address class imbalance. SMOTE generates synthetic samples of the minority class (anemia) to balance the dataset and improve classifier performance.

5. Hyperparameter Tuning: Grid search, a hyperparameter tuning technique, is utilized to optimize the parameters of the Random Forest classifier. Grid search systematically explores a range of hyperparameter values to find the combination that yields the best performance.

6. Cross-Validation: To evaluate the model's performance and ensure its robustness, 10-fold cross-validation is employed. The dataset is divided into 10 equal-sized folds, with the model trained on 9 folds and evaluated on the remaining fold. This process is repeated 10 times to obtain reliable performance estimates.

7. Model Evaluation: The performance of the Random Forest classifier is evaluated using standard metrics such as accuracy, precision, recall, and F1-score. Visualization techniques are used to analyze the model's predictions and identify any areas for improvement.

Overall, this individual project aims to develop an effective and accurate AI-based solution for anemia detection using conjunctiva images. By leveraging machine learning algorithms and rigorous evaluation methods, the project seeks to contribute to early diagnosis and intervention in individuals at risk of anemia.

Accuracy on Test Data: 0.9047619047619048
Classification Report:
               precision    recall  f1-score   support

          No       0.94      0.94      0.94        16
         Yes       0.80      0.80      0.80         5

    accuracy                           0.90        21
   macro avg       0.87      0.87      0.87        21
weighted avg       0.90      0.90      0.90        21

Mean Cross-Validation Accuracy: 0.848076923076923
