#Importing the libraries

import pandas as pd  #Managing datasets
import numpy as np  #Mathematical Library
import matplotlib.pyplot as plt  #For plotting models

#Import the datasets
#Make sure you set the correct working directory

dataset = pd.read_csv("Data.csv")
print dataset #Check if correctly loaded

#Distinguish the features from the dependent variable

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

print X, y  #Check the splitting   

#What if we have missing data ?
# Mean of that particular column

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# axis = 0 -> column, axis = 1 -> row
#Now we will fit the imputer object to feature matrix

imputer = imputer.fit(X[:, 1:3]) #upper bound is excluded
X[:, 1:3] = imputer.transform(X[:, 1:3]) #Apply the Imputer model and save it in the original feature map
print X

#Let's encode the categorical data
#Encode Country and Purchased columns

#Import the library

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_country = LabelEncoder()
X[ :, 0] = label_country.fit_transform(X[:,0])  #Encoded the country columns
print X

#Here encoding problem arises
#OneHotEncoding

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
print X
# Encoding the Dependent Variable
labelencoder_purchased = LabelEncoder()
y = labelencoder_purchased.fit_transform(y)
print y

#Let's spit the dataset
#import library

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 






