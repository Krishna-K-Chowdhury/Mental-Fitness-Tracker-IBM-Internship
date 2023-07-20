# Mental Health Fitness Tracker
The Mental Health Fitness Tracker project is dedicated to analyzing and predicting the mental fitness levels of individuals from different countries, each dealing with unique mental disorders. By utilizing regression techniques, the project gains valuable insights into mental health while making reliable predictions based on the available data. Alongside these functionalities, the project also provides users with a practical platform to track their mental health and fitness levels. Built using Python, it ensures efficiency and effectiveness in achieving its objectives.

## Installation
To use the code and execute samples, follow these steps:

1. Ensure that you have Python 3.x installed on your system.
2. Install the required libraries by running the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly.express
```
    
3. Download the project files and navigate to the project directory.

## Contributing
Contributions enrich the open-source community, fostering learning and inspiration. Your efforts are sincerely appreciated.

1. Fork the repo
2. Clone the project
3. Create your feature branch
4. Commit your changes
5. Push to the branch
6. Open a pull request


## References
- Datasets that were use in here were taken from [KAGGLE](https://www.kaggle.com/datasets/programmerrdai/mental-health-dataset)
- This project was made during my internship period for [Edunet Foundation](https://edunetfoundation.org) in association with [IBM SkillsBuild](https://skillsbuild.org) and [AICTE](https://internship.aicte-india.org)

## PYTHON CODE

# OPEN IN COLAB
```bash
Please note, you have to mount the directory containing the datasets in your drive
```
```bash
Link to google colab: https://colab.research.google.com/drive/1MIop9j07C209B4cfQp7pelAOQSZ5O6GN?usp=sharing
```

# IMPORT REQUIRED LIBRARIES
```bash
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
```

# READING DATASETS
```bash
df1 = pd.read_csv("prevalence-by-mental-and-substance-use-disorder.csv")
df2 = pd.read_csv('mental-and-substance-use-as-share-of-disease.csv')
```

# SHOW DATA SET
```bash
df1.head()
```
```bash
df2.head()
```

# MERGING TWO DATASETS
```bash
data = pd.merge(df1, df2)
data.head()
```

# DATA CLEANING
```bash
data.isnull().sum()
```
```bash
data.drop('Code', axis=1, inplace=True)
```
```bash
data.size,data.shape
```

# RENAMED COLUMNS 
```bash
data.set_axis(['Country','Year','Schizophrenia', 'Bipolar_disorder', 'Eating_disorder','Anxiety','drug_usage','depression','alcohol','mental_fitness'], axis='columns', inplace=True)
data.head()
```

# EXPLORATORY ANALYSIS
```bash
plt.figure(figsize=(12,6))
sns.heatmap(data.corr(),annot=True,cmap='winter')
plt.plot()
```

```bash
sns.jointplot(data,x="Schizophrenia",y="mental_fitness",kind="reg",color="r")
plt.show()
```
```bash
sns.jointplot(data,x='Bipolar_disorder',y='mental_fitness',kind='reg',color='g')
plt.show()
```
```bash
sns.jointplot(data,x='Eating_disorder',y='mental_fitness',kind='reg',color='y')
plt.show()
```
```bash
sns.jointplot(data,x='Anxiety',y='mental_fitness',kind='reg',color='b')
plt.show()
```
```bash
sns.jointplot(data,x='drug_usage',y='mental_fitness',kind='reg',color='m')
plt.show()
```
```bash
sns.jointplot(data,x='depression',y='mental_fitness',kind='reg',color='orange')
plt.show()
```
```bash
sns.jointplot(data,x='alcohol',y='mental_fitness',kind='reg',color='grey')
plt.show()
```
```bash
sns.pairplot(data,corner=True)
plt.show()
```
```bash
mean = data['mental_fitness'].mean()
mean
```
```bash
fig = px.pie(data, values='mental_fitness', names='Year')
fig.show()
```
```bash
fig=px.bar(data.head(10),x='Year',y='mental_fitness',color='Year',template='ggplot2')
fig.show()
```

# VARIATIONS IN MENTAL FITNESS OF DIFFERENT COUNTRIES (YEARWISE)
```bash
fig = px.line(data, x="Year", y="mental_fitness", color='Country',markers=True,color_discrete_sequence=['red','blue','white','lightskyblue'],template='plotly_dark')
fig.show()
```
```bash
df=data.copy()
df.head()
```
```bash
df.info()
```

```bash
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
for i in df.columns:
    if df[i].dtype == 'object':
        df[i]=l.fit_transform(df[i])

X = df.drop('mental_fitness',axis=1)
y = df['mental_fitness']
```
```bash
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=2)
```
```bash
X = df.drop('mental_fitness',axis=1)
y = df['mental_fitness']
```
```bash
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=2)
```
```bash
print("xtrain: ", xtrain.shape)
print("xtest: ", xtest.shape)
print("ytrain: ", ytrain.shape)
print("ytest: ", ytest.shape)
```

# LINEAR REGRESSION

```bash
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
lr = LinearRegression()
lr.fit(xtrain,ytrain)

# model evaluation for training set
ytrain_pred = lr.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = (np.sqrt(mean_squared_error(ytrain, ytrain_pred)))
r2 = r2_score(ytrain, ytrain_pred)

print("Model performance for training set: ")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
ytest_pred = lr.predict(xtest)
mse = mean_squared_error(ytest, ytest_pred)
rmse = (np.sqrt(mean_squared_error(ytest, ytest_pred)))
r2 = r2_score(ytest, ytest_pred)
percentage = "{:.2%}".format(r2)

print("Model performance for testing set: ")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print('Percentage is {}'.format(percentage))
```

# DECISION TREE REGRESSOR

```bash
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(xtrain,ytrain)

# model evaluation for training set
ytrain_pred = dtr.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = (np.sqrt(mean_squared_error(ytrain, ytrain_pred)))
r2 = r2_score(ytrain, ytrain_pred)

print("Model performance for training set: ")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
ytest_pred = dtr.predict(xtest)
mse = mean_squared_error(ytest, ytest_pred)
rmse = (np.sqrt(mean_squared_error(ytest, ytest_pred)))
r2 = r2_score(ytest, ytest_pred)
percentage = "{:.2%}".format(r2)

print("Model performance for testing set: ")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print('Percentage is {}'.format(percentage))
```

# RANDOM FOREST REGRESSOR

```bash
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(xtrain, ytrain)

# model evaluation for training set
ytrain_pred = rf.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = (np.sqrt(mean_squared_error(ytrain, ytrain_pred)))
r2 = r2_score(ytrain, ytrain_pred)

print("Model performance for training set: ")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
ytest_pred = rf.predict(xtest)
mse = mean_squared_error(ytest, ytest_pred)
rmse = (np.sqrt(mean_squared_error(ytest, ytest_pred)))
r2 = r2_score(ytest, ytest_pred)
percentage = "{:.2%}".format(r2)

print("Model performance for testing set: ")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print('Percentage is {}'.format(percentage))
```

# Predicting Mental Fitness using Random Forest Model.

```bash
np.random.seed(range(0, 100))
print("Welcome to Mental Fitness Tracker!\n")
country = l.fit_transform([input("Enter the country name: ")])
year = int(input("Enter the year: "))
schizophrenia = (float(input("Enter the Schizophrenia rate in %: ")))/100
bipolar_disorder = (float(input("Enter the Bipolar Disorder rate in %: ")))/100
eating_disorder = (float(input("Enter the Eating Disorder rate in %: ")))/100
anxiety = (float(input("Enter the Anxiety rate in %: ")))/10
drug_usage = (float(input("Enter the Drug Usage rate in %: ")))/100
depression = (float(input("Enter the Depression rate in %: ")))/10
alcohol = (float(input("Enter the Alcohol rate in %: ")))/100

data = [[country, year, schizophrenia, bipolar_disorder, eating_disorder, anxiety, drug_usage, depression, alcohol]]
data = pd.DataFrame(data, columns=['Country', 'Year', 'Schizophrenia', 'Bipolar_disorder', 'Eating_disorder', 'Anxiety', 'drug_usage', 'depression', 'alcohol'])

data['mental_fitness'] = rf.predict(data)
print("Mental fitness score in percent is: ", (data['mental_fitness'][0])*10)
```

# CONCLUSION

The mental fitness of a person is dependent on various factors like schizophrenia, bipolar disorder, eating disorder, anxiety, drug usage, depression, alcohol, etc.
Here, we have used the Random Forest Regressor model to predict the mental fitness based on the above factors. The model has an accuracy of 99.99% which is quite good. The model can be used to predict the mental fitness. The model can be further improved by using more data and by applying other algorithms.
