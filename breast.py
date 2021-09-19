#MADE BY MICHAEL JOSPEH KIBET 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import warnings
import os 
warnings.filterwarnings("ignore")
import datetime


#loading the dataset
data=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')


data.head()      #displaying the head of dataset


data.describe()      #description of dataset 


data.info()


data.shape       #569 rows and 33 columns





data.value_counts


data.dtypes

data.isnull().sum()



data.drop('Unnamed: 32', axis = 1, inplace = True)


data


#visualizing the data / showing 

data.corr()


plt.figure(figsize=(18,9))
sns.heatmap(data.corr(),annot = True, cmap ="Accent_r")

sns.barplot(x="id", y="diagnosis",data=data[160:190])
plt.title("Id vs Diagnosis",fontsize=15)
plt.xlabel("Id")
plt.ylabel("Diagonis")
plt.show()
plt.style.use("ggplot")


sns.barplot(x="radius_mean", y="texture_mean", data=data[170:180])
plt.title("Radius Mean vs Texture Mean",fontsize=15)
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.show()
plt.style.use("ggplot")




mean_col = ['diagnosis','radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

sns.pairplot(data[mean_col],hue = 'diagnosis', palette='Accent')




sns.violinplot(x="smoothness_mean",y="perimeter_mean",data=data)


plt.figure(figsize=(14,7))
sns.lineplot(x = "concavity_mean",y = "concave points_mean",data = data[0:400], color='green')
plt.title("Concavity Mean vs Concave Mean")
plt.xlabel("Concavity Mean")
plt.ylabel("Concave Points")
plt.show()


worst_col = ['diagnosis','radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

sns.pairplot(data[worst_col],hue = 'diagnosis', palette="CMRmap")



#TIME TO TRAIN THE AND TEST THE DATA 

# Getting Features

x = data.drop(columns = 'diagnosis')

# Getting Predicting Value
y = data['diagnosis']


#train_test_splitting of the dataset
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)



print(len(x_train))
print(len(x_test))
print(len(x_test))
print(len(y_test))
