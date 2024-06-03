# Project_4: Building Machine Learning Models for Heart Patients and predicting which Model is better

**Overview and Purpose**

We have chosen data for 299 patients who had heart disease. This data is collected during the follow-up period and try to predict if the patient die or not depending on their medical history with certain features.

**Key Objectives:**

1. Data Exploration: Dive into open source data from UCI Machine Learning Repository which is  collection of databases, domain theories, and data generators that are used by the machine learning community for the empirical analysis of machine learning algorithms.
2. Feature Importance and Relation: There are total 14 features given for each patient including (Anaemia, High blood pressure, Creatinine phosphokinase, Diabetes, Ejection fraction, Platelets, Sex, Serum creatinine level, Serum sodium level, Smoking, Time, DEATH_EVENT) and predicting death_events and understand which factors are most predictive of the outcome. 
3. Machine Learning Model: Building 2 machine learning models - Random Forest Classifier and Logistic Regression while comparing and predicting the death_event for the patients who has certain history during their visits to the doctor and suggesting which is the better model.
4. Interpretation: 
 

**Team Members:**

  	Justen Hix
   
  	Payal Bansal
  
   	Sheena Picket

**Ethical Considerations**

**Data Source**

We have used Open source data from (https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records).
This dataset contains the medical records of 299 patients from Cleveland, Hungary, Switzerland, and the VA Long Beach who had heart failure, data collected during their follow-up period, where each patient profile has 13 clinical features.


**Technologies Used**

1. Programming Language: Python
2. Libraries and Modules: 
	- Pandas: For data manipulation and analysis 
	- Spark SQL: For data retrieval from csv file.
	- NumPy: For numerical operations and array manipulations 
	- Plotly and Ployly.Express: Library used to create visualizations
   	- Seaborn:  Used to visualize distributions
	- sklearn: For RandomForestClassifer, Logistic Regression model, Confusion matrix, classification report, accuracy score
3. Integrated Development Environment (IDE): Jupyter Notebook: Used for writing, testing, and debugging the Python script.
6. Version Control: Git: Used for version control and collaborative development.
7. Database: Spark SQL 
8. Project Documentation: README.md: For project documentation and instructions.

**Data Cleaning, Model creation and Prediction Process**

1. Do the required imports, initialize the 'findspark', create a Spark session and read the csv file and store it in a DataFrame.
2. Data Preprocessing that includes Feature and Target separation. Here 'DEATH_EVENT' is our target variable and rest are selected as features.
3. Converting the Spark DataFrame in Pandas DataFrame and split the dataset into training and testing datasets.
4. Then we create a Standard Scaler instance, fit it on training dataset, scaled the data and create a Random Forest Classifier and fit that into training data.
5. Make prediction for the testing dataset and finally calculate accuracy score and generate the confusion matrix for the model.
6. Calculate the feature importance and create a bar graph to visually see the results. 
7. Next we create a Logistic Regression model, fit the model onto training dataset and getting the score for training and testing datasets.
8. We make the predictions on testing data and finally get the accuracy score for model. We create the confusion matrix and classification report for the model.
9. We create some bar graphs for features having Boolean values and scatter plots for other feature with 'DEATH_EVENT'.
10. Create few Box Plot for some features with 'DEATH_EVENT'.

**Model Optimization**

1. We have checked if there are any missing values and then delete if there are any.
2. We run are Random Forest Model again but the accuracy comes out exact same
3. Secondly, we used ross-validation technique which helps to ensure that the model's performance is not just good on the training data but also on unseen data.
4. The mean Cross-Validation Accuracy comes out to be 83.9% which indicates the model to be robust and not overly sensitive to different subsets of the data.

![image](https://github.com/justenhix/Project4Group1/assets/148804724/73ca6e2c-f368-43b1-9d75-c120db94074e)


**Results**

1. Confusion Matrix for Random Forest Classifier

 ![image](https://github.com/justenhix/Project4Group1/assets/148804724/8cdc89c1-1905-425e-ad63-48482bada91e)

2. Feature importance Bar Graph

![image](https://github.com/justenhix/Project4Group1/assets/148804724/4dc75120-f07f-44d7-ba60-267f63a1fe4a)

3. Confusion Matrix for Logistic Regression

![image](https://github.com/justenhix/Project4Group1/assets/148804724/c8d3fa6c-ea32-455c-ba00-5525f0588bf6)

4. Plot of Death events for patients with their Anemia Status

![image](https://github.com/justenhix/Project4Group1/assets/148804724/9d23b6b4-58b7-481c-8442-35a8c33ee7e6)

5. Plot of Death events for patients w.r.t to Diabetes(sugar level)

![image](https://github.com/justenhix/Project4Group1/assets/148804724/2c6c438f-cfd1-4eb2-a9af-592128b6b1ae)

6. Scatter plot for age vs creatinine_phosphokinase(help diagnose and monitor various conditions related to muscle damage, heart conditions, and other medical issues.)

![image](https://github.com/justenhix/Project4Group1/assets/148804724/26b887fd-a8bb-43d3-88ff-311c99129c46)

7. Scatter plot for age vs platelets(small blood cells that help stop bleeding by forming clots)

![image](https://github.com/justenhix/Project4Group1/assets/148804724/4281b66d-a99a-4cfd-97d7-81f12524790a)

8. Box Plot of Ejection Fraction vs. DEATH_EVENT

![image](https://github.com/justenhix/Project4Group1/assets/148804724/821020d2-51cc-4292-9fe2-5cef60be2b1e)

9. Box Plot of Serum Creatinine vs. DEATH_EVENT

![image](https://github.com/justenhix/Project4Group1/assets/148804724/e08444ae-6575-4797-9d36-e16a4ccb99a8)




**Conclusions**

While we compare the Random Forest Classifier model with Logistic Regression model, we can see that Model1 has slightly higher accuracy as compared to Model2. Also, Model 1 performs better overall, particularly in terms of accuracy and the performance metrics for class 1 (Comorbidity). So Model1 is more preferred due to its higher accuracy and classification matrices.


**Implications/Limitations**

1. Feature Limitation: The dataset contains only 14 features whereas it is indicated that there are total 76 attributes which are required to accurately diagnose the heart disease. Due to limited features there are chances we might not capture all aspects that might be important. Important features such as genetic information, lifestyle factors, and detailed medical history are not included.

2. Static Nature: The dataset is static and does not account for temporal changes or trends in patient health status over time. This limits the ability to develop models that can handle time-series data or predict the progression of heart disease.

3. Lack of Demographic Diversity: The dataset may not be representative of the broader population, as it could be biased towards certain demographics, such as age, gender, or ethnicity. This limits the ability of the model to generalize to different populations.

4. Small Sample Size: The dataset contains a relatively small number of samples, which may limit the generalizability of the models trained on it. Small datasets can also lead to overfitting, where the model performs well on the training data but poorly on new, unseen data.




