# Building a Machine Learning Pipeline to Flag Students at Risk of Failing 

## Overview
In this project, I will be posing as an AI Engineer working for the Ministry of Education. 

The Ministry of Education is currently developing a resource Secondary School Mathematics Head-of-Deaprtment can use to flag students at risk of failing their Mathematics O-Level exam based on their environmental factors.

I have been given data on the previous year's country-wide O-Level mathematics results, alongside details about each participant's background, and am expected to design an end-to-end machine learning pipeline that can predict whether a student will fail their O-Level Mathematics exam.

## Data Dictionary
The dataset contains the following columns:

1. **student_id**: Unique ID for each student
2. **number_of_siblings**: Number of siblings
3. **direct_admission**: Mode of entering the school
4. **CCA**: Enrolled CCA
5. **learning_style**: Primary learning style
6. **tuition**: Indication of whether the student has a tuition
7. **n_male**: Number of male classmates
8. **n_female**: Number of female classmates
9. **gender**: Gender type
10. **age**: Age of the student
11. **hours_per_week**: Number of hours student studies per week
12. **attendance_rate**: Attendance rate of the student [It is unclear what the time frame is; whether it is for the last year or last year before completing O-Levels, etc] (%)
13. **sleep_time**: Daily sleeping time (hour:minutes)
14. **wake_time**: Daily waking up time (hour:minutes)
15. **mode_of_transport**: Mode of transport to school
16. **bag_color**: Colour of student’s bag
17. **final_test**: Student’s O-level mathematics examination score.

While 'final_test' was stated as the y-variable in the data dictionary provided, I will be engineering a new column 'failed', which will flag students who failed the exam.

## Notes on Data Cleaning and Feature Engineering

1. I dropped 900 rows which were duplicates of data in the original dataframe.
2. I dropped all null values under the features 'final_test' and 'attendance_rate'. The are very important variables – the former because of its nature as the 'y-variable', and the latter as a potentially significant influence on 'final_test'. However, there were no clear clues on how I could impute the null values, so I dropped rows where these columns were null, altogether amounting to less than 8% of the dataset.
3. I engineered 'failed', a new column standing in as the y-variable, which flags 1.0 for students who failed their exam, and 0.0 for students who passed.
4. I imputed the null values under 'CCA' - which comprised 24% of the column - using random forest regression, based on the other X_variables in the dataset. I did this as it was a potentially influential variable on 'failed', and I wanted to keep 'CCA' in the dataset.
5. I decided to drop the following features as they were either arbitrary or irrelevant to the prediction: 'age', 'bag_color', student_id'
6. I engineered 'sleep_dur', which reflects the duration for which students slept daily, out of 'sleep_time' and 'wake_time'.

## Notes on Exploratory Data Analysis (EDA)
My EDA has uncovered the following findings about students who failed their exam:

1. There are more females than males among those who failed
2. Most of those who attended less than 90% of lessons ended up failing
3. Most of those who only spent 0-4 hours studying math per week ended up failing
4. Almost all who failed have at least 1 sibling
5. Most students who slept less than 7 hours ended up failing

These observations signalled towards the importance of the following features to the outcome of 'failed': *'gender', 'attendance_rate', 'hours_per_week', 'number_of_siblings', 'sleep_dur'*

While investigating inter-variable correlations, I found strong multicolinearity between 'sleep_time', 'sleep_dur' and 'attendance_rate'. **Among the three, 'sleep_time' was the least influential on 'failed' (after checking the Pearsons correlation values between the X-variables and 'failed'), so I decided to drop it.**

Additionally, the p-values from the chi-squared test of independence between 'mode_of_transport' and other categorical variables as well as with 'failed' was high, indicating that 'mode_of_transport' may was not very influential on the other X-variables or on 'failed'. **As a result, I dropped 'mode_of_transport as well.**

Finally, **I decided to drop 'wake_time'**, as it does not encompass a wide range, and the Pearson correlation between 'wake_time' and 'failed' suggests that there is close to no linear relationship between 'wake_time' and 'failed'.

The final dataset has the following features:
1. gender
2. learning_style
3. attendance_rate
4. hours_per_week
5. n_male
6. n_female
7. direct_admission
8. CCA
9. tuition
10. number_of_siblings
11. failed
12. sleep_dur

## Pipeline Outputs and Modelling Results
I used a logistic regression model because I wanted to ensure that the model was interpretable to as many people as possible. 

Using my pipeline will output the following:

1. A dataset, 'predict_df.csv', which contains
- The predicted y values ('predicted_failed')
- The actual y values ('actual_failed') so the HOD can see how the predictions compare to the acutal values
- The X-variables from the test set (X_test), so the HOD has more context by which to be able to evaluate the predicted y values for themselves

2. An array of metrics which evaluate the model's performance, namely accuracy, precision, recall, and a confusion matrix. 

The results of my modelling pipeline is as follows:
- Accuracy (How accurate the model's predictions is overall) = 94.3%
- Precision (Out of all predicted failures, which were actually failures?) = ~86%

## What Can We Learn from the Data?
- Schools can encourage better daily time management among students, to ensure they get adequate  rest (>= 7 hours/day) and enough time to study math >= 5 hours/week
- While almost all who failed tended to have at least one sibling, more data should be collected to understand the relationship between 'number_of_siblings' and 'failed' better (e.g. do such students need to fulfil a caregiving role to younger siblings at home?)

## Limitations
- I had to make a couple of assumptions about the data:

1. The students in the dataset are Secondary 4 students (age range in the dataset is between 15.0-16.0)
2. The students’ background information was representative of their lifestyle/environmental factors until the O-Level exam

- Analysis was limited by the data presented to me, which lacks data on factors that are potentially influential on 'failed' (e.g. household income, neighbourhood, amount of after-school enrichment, etc)

- While the use of a logistic regression model translates to better interpretability by School HODs, such models are unable to detect non-linear relationships between dependent and independent variables or between dependent variables.
