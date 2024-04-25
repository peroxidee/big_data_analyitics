
import pandas as pd
import numpy as np
import nltk as nltk
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
matplotlib.use('TkAgg')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        
df = pd.read_csv('COVID-19 Survey Student Responses.csv')


print(df.dtypes)
rows, columns = df.shape
print(f"The dataset contains {rows} rows and {columns} columns.")
missing_values = df.isnull().sum()
print(missing_values)


categorical_columns = ["Region of residence", 'Rating of Online Class experience', 'Medium for online class']
for column in categorical_columns:
    if df[column].isnull().sum() > 0: 
        mode_value = df[column].mode()[0]
        df[column] = df[column].fillna(mode_value)
        
numerical_columns = ['Age of Subject', 'Time spent on Online Class','Time spent on self study', 'Time spent on fitness','Time spent on sleep','Time spent on social media']

for column in numerical_columns:
    if df[column].isnull().sum() > 0:
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)
        

missing_values = df.isnull().sum()
print(missing_values)


inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(df[['Age of Subject', 'Time spent on Online Class']])
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()


kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df[['Age of Subject', 'Time spent on Online Class']])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age of Subject', y='Time spent on Online Class', hue='Cluster', palette='viridis')
plt.title('Student Clusters')
plt.show()

data_filtered = df[df['Rating of Online Class experience'] != 'NA']
X = data_filtered[['Age of Subject', 'Time spent on self study', 'Time spent on fitness', 'Time spent on sleep']]
# variable encoding to transfer over a qualitative variable to a quantitaive varible.
y = data_filtered['Rating of Online Class experience'].map({'Excellent':4,'Good':3,'Average':2,'Poor':1,'Very Poor':0})