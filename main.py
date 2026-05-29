import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())


df['Age'].fillna(df['Age'].mean())
df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Cabin'].fillna(df['Cabin'].mode()[0])

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.savefig("survival_by_gender.png")
plt.show()


scaler = StandardScaler()

df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

sns.boxplot(x=df['Fare'])
plt.title("Fare Boxplot")
plt.savefig("boxplot_before_outliers.png")
plt.show()

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df['Fare'] >= lower) & (df['Fare'] <= upper)]

# after remove outliers 
sns.boxplot(x=df['Fare'])
plt.title("Fare Boxplot")
plt.savefig("boxplot_after_outliers.png")
plt.show()

print("Final Shape:", df.shape)
print("Preprocessing Completed")