# Solubility Prediction by Random Forest Algorithm Using RDKit 2D and 3D Descriptors  
**Mehran Mansouri, Pharmacy student at Mashhad University of Medical Sciences**  

## Important notes:
- This project is just a training project on RDKit 2D and 3D Descriptors. So, this is not a functional project.
- Make sure to install miniconda and RDKit. Open this in conda kernel.
- 
## Abstract
This project involves building a predictive machine learning model using molecular descriptor data. The workflow includes data preprocessing with scaling, dimensionality reduction using PCA, and model training with hyperparameter optimization via GridSearchCV. PCA reduces the feature set from 221 to 17 components, and the best model is selected based on cross-validation performance. Despite these efforts, the test R² is lower than expected, prompting further exploration of alternative feature selection methods, model experimentation, and regularization techniques to enhance predictive accuracy.


## Details
This project focuses on building a machine learning model to predict a target variable using molecular descriptor data. The workflow includes data preprocessing, feature selection, dimensionality reduction, and model training with hyperparameter optimization. Below is a detailed breakdown of the steps:

1. Create database:  
I used some random SMILEs received from an AI model as my data. Not all of these data were valid. Then, I added labels considering logP, molecular weight, and hydrogen bond donors.

2. Data Preprocessing:  
The dataset contains molecular descriptors as features and a target variable for prediction.
The data is split into training and testing sets (X_train, X_test, y_train, y_test).
Scaling is applied to the features using RobustScaler to normalize the data. This ensures that all features contribute equally to the model and avoids issues caused by varying feature scales or outliers.

3. Dimensionality Reduction:  
Principal Component Analysis (PCA) is used to reduce the number of features from 221 to 10. PCA identifies the directions (principal components) that capture the most variance in the data, allowing for a more compact representation of the dataset.
PCA is fit on the scaled training data (X_train_scaled) and then applied to both the training and testing sets to ensure no data leakage.

4. Model Training and Hyperparameter Optimization:  
A machine learning model is trained using the reduced feature set (X_new_train and X_new_test).
Hyperparameter optimization is performed using GridSearchCV to find the best parameters for the model.
The best model is selected based on cross-validation performance, and its parameters are printed for reference.

5. Evaluation:  
The model is evaluated on the test set using the R² metric to assess its predictive performance.
Despite using the best parameters and reducing the feature set, the test R² is lower than expected, indicating potential issues such as overfitting, short data size, or suboptimal feature selection.

**Suggestions for Improvement:**  
Alternative feature selection methods (e.g., SelectKBest, RFE) are recommended to identify features most relevant to the target variable.
Experimentation with different models (e.g., Random Forest, Gradient Boosting) and ensemble methods is suggested to improve performance.
Regularization techniques (e.g., Ridge, Lasso) and hyperparameter tuning are proposed to address overfitting and improve generalization.

## Get in contact with me:
- Email and Skype: Mehran.mansouri811@gmail.com
- Phone number: +989150511552
- Telegram: @Mehran_mns

