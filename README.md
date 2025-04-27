**Predicting 10-Year Coronary Heart Disease Risk Using Machine Learning**


**Abstract**
Coronary heart disease (CHD) is one of the top causes of death globally. Early prediction of risk associated with CHD provides the opportunity for, and promotes, preventive activities, ultimately reducing long-term health complications. This study established Support Vector Machine (SVM) models to predict 10-year risk associated with CHD, utilizing clinical and behavioral data from the Framingham Heart Study. Each SVM model was conducted using systematic feature selection, systematic data preprocessing, systematic handling of class imbalance, and systematic evaluation of models. Results indicate that the model recall for high-risk patients significantly improved with balancing techniques, rendering the model useful from a clinical perspective.


**1.	Introduction**
Heart disease continues to be the number one killer in the world, accounting for around 18 million deaths each year (World Health Organization, 2023). If it is possible to identify people who are at risk of developing coronary heart disease (CHD), they can be accessed early enough to provide effective interventions. Traditional statistical models commonly used such as the Framingham Risk Score (FRS) are commendable but also limit researchers to pre-defined uses. This study uses machine learning, namely Support Vector Machine (SVM), to model predicted risk of CHD using the Framingham Heart Study dataset after 10 years.

**2.	Literature Review**
A number of studies have shown that machine learning models may be able to enhance clinical risk predictions. For example, Kavakiotis et al. (2017) argued that the use of SVMs, random forests, and other machine learning models are very effective models of healthcare analytics. In addition, Siontis et al. (2018) affirm that ML-based risk scores are typically better than statistical risk scores in the area of cardiovascular predictions. Chicco and Jurman (2020) reiterated that when tuned correctly, SVMs outperform other models for obtaining high precision and recall in biomedical datasets, especially with imbalanced data.  Not only that, but ensemble models such as Random Forest (Breiman, 2001) and ensemble learning methods like boosting (XGBoost; Chen & Guestrin, 2016) reduce variance and bias that often support improved prediction robustness. Other practices, dealing with class imbalance (such as applying SMOTE (Chawla et al. 2002)), are now often standard when looking to predict medical outcomes in datasets that contain many more instances of non-minority outcomes, although risk predicting the minority outcome (such as heart attack) is extremely important.

**3.	Methods**


   ![image](https://github.com/user-attachments/assets/d13d3a92-1012-45d5-99c9-7749fd347ece)


Figure 1: Project Workflow
                     
**3.1 Dataset**


![image](https://github.com/user-attachments/assets/c1295269-b6a3-45cb-9b23-2dde60c3f801)


Figure 2: Framingham Heart Study dataset
                   
The dataset being used is the Framingham Heart Study dataset which contains 4,240 records of patients who have attributes such as age, gender, cholesterol levels, blood pressure, glucose, smoking habits, and medical history. The target variable in this case is TenYearCHD, which is whether the patient developed CHD in the next 10 years.

**3.2 Data Preprocessing**
We found missingness in the features, for each feature with missingness, we used the mode to impute the missingness. Feature scaling was performed using StandardScaler to standardize features to mean 0, and standard deviation 1.


 ![image](https://github.com/user-attachments/assets/1635f933-ac45-4d65-9ae8-713a8056e2a9)


Figure 3: Missing values

**3.3 Feature Selection**
Using the ANOVA F-test via SelectKBest for feature selection, we found the following top 8 predictors of CHD: age, systolic blood pressure (sysBP), prevalent hypertension, diastolic blood pressure (diaBP), glucose, diabetes, gender (male), and BPMeds.


 ![image](https://github.com/user-attachments/assets/8fa90f59-fbb0-4682-ba70-7e846344a870)


Figure 4: Best Features

**3.4 Train-Test Split**
The dataset was split into an 80% training set and 20% testing set using stratified sampling to preserve the class distribution.

**3.5 Handling Class Imbalance**
Taking into account the severe imbalance in the dataset (ca. 15% positive CHD cases), three approaches were investigated:
1)	Naive SVM without addressing imbalance
2)	Class Weighting in SVM class_weight='balanced'
3)	SMOTE (synthetic minority oversampling technique) used to synthetically oversample in the case of minority class cases
The goal here is to make use of the positive labels for CHD cases without biasing the SVM towards the large number of majority positive labels the determine the majority case.

**3.6 Model Building**
A Support Vector Machine (SVM) was trained with a linear kernel using three different proposed configurations:
1)	Naive SVM: No special class imbalance handling.
2)	SVM with Class Weights: This configuration modified the loss function to easier penalize mistakes on the minority class.
3)	SVM with SMOTE: This configuration oversampled the minority class to create a balanced dataset before training the model.


![image](https://github.com/user-attachments/assets/242af8e3-04df-4586-8ba2-9b7b06bc5fad)


Figure 5: Smote Data Balancing

The data set was split using stratified sampling 80 -20 for training and testing. The same test set was used to evaluate all models.


 ![image](https://github.com/user-attachments/assets/2807e4b2-61b8-45f8-894b-468e18abe1a5)


Figure 6: Naive SVM



 ![image](https://github.com/user-attachments/assets/2587a35a-c9ae-46f3-a8f7-8c757e55d205)


Figure 7: SVM with class weight balanced



![image](https://github.com/user-attachments/assets/a02a596d-e40e-4045-8444-8ac1be6da5cb)


Figure 8: SVM using SMOTE

**4.	Results**
Method	           Accuracy	 Recall(CHD)  ROC AUC
SVM (No balancing)	  85%	       0.00	   0.68
SVM + Class Weights	  65%	       0.64	   0.69
SVM + SMOTE	          64%	       0.64	   0.69


**4.1 Interpretation of Results**
Naive SVM produces a nice accuracy of 85% because it was just predicting the dominant class of No CHD most of the time. It also had 0% recall for CHD, which made it almost worthless. SVM with Class Weights decreased significantly to 65% accuracy, but at least maintained a respectable recall of 64% CHD, making it much more clinically useful.  SVM with SMOTE performed similarly as SVM with Class Weights with 64% accuracy and 64% recall. This also showed that synthetic oversampling does effectively train the model to identify the minority class cases.  Although the balance models are slightly lower in overall accuracy compared to the naive models, they remarkably increased sensitivity (recall) for CHD which is the main clinical aim.

**5.	 Model Performance Analysis**
These findings illustrate that accuracy alone is a misleading metric when evaluating the performance of a model built on an imbalanced dataset.


![image](https://github.com/user-attachments/assets/6a90c7b5-ddfe-4d2f-b70c-b16477849cb8)


 ![image](https://github.com/user-attachments/assets/b417becf-2aa5-4a6c-8fa4-63098598444d)


Figure 9: ROC curve for the results

Without balancing, the model learned to predict the majority class (No CHD) to achieve the best accuracy but ignored serious cases of CHD. The implementation of class weights altered the learning ability of the model to be more sensitive, as it would over penalize misclassification of samples that belonged to the minority class.  Additionally, the SMOTE technique improved the model with its synthetic CHD samples to force the model to learn a more balanced decision boundary.  In both cases (with and without class weights and SMOTE), there was not a significant change in performance, which is expected.  SMOTE offered something class weights did not to diversify the dataset, the addition of more theoretical examples around examples of minority class members that could potentially be used for later deployment for when it is required to interpret and deploy a model to newer data that has not yet been seen. Thus, the SVM with SMOTE is the preferred methods towards achieving a more robust, generalizable model for use in applications in the real-world.

**6.	Conclusion**
We developed a Support Vector Machine (SVM) model to predict 10-year coronary heart disease (CHD) risk using data from the Framingham Heart Study. The first SVM model was a naive SVM which provided a 85% accuracy. However, it was unable to identify any positive cases of CHD, which highlighted the issues associated with relying solely on accuracy when working with imbalanced datasets. To partially mitigate this issue, we utilized two balancing strategies: class weighting and SMOTE. Both balancing strategies improved the recall for CHD patients to 64% while the ROC AUC stayed at approximately 0.69. As a result of the balancing approaches, the naive SVM accuracy fell to approximately 64-65%. The two SVMs were clinically relevant in that it focused more on correctly identifying high-risk patients, rather than overall accuracy. The SVM using SMOTE was able to add more robustness by augmenting the training data, which may allow it to be more suitable for practical use in the real world. This work has shown that in the correct hands, with pre-processing of the data, feature selection, and handling of imbalance, an SVM may have adequate performance and assist in identifying risks for CHD at an early stage.

**7.	 Future Work**
Future directions will also include obtaining better models with the implementation of higher-level methods. In this approach machine learning will include ensemble learning methods such as Random Forest and XGboost to start modelling the complex relationships that are present in the data. Hyperparameter optimization will be conducted with GridSearchCV and other methodologies until better models with optimal hyperparameters lead to better generalization. All models would be validated by cross validations to measure reliability and reproducibility across multiple data sets. Further to improving models I also see a simple web-based clinical app being produced that will allow any health care provider to enter into the characteristics of a patient and obtain an instantaneous prediction of a patient’s risk of CHD. This would help create a direct pathway from research models to real world practical and preventative health care use.

 
![image](https://github.com/user-attachments/assets/7d0ef74e-8d4c-4ea2-958d-6e1b35a6da4c)


Figure 10:Bagging and Boosting


















