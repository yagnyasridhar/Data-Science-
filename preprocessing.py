import numpy as nu
import matplotlib.pyplot as pt
import pandas as pnd
from sklearn.impute import SimpleImputer as si 

'''
Extraction
'''
dataset = pnd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
#print(X)
#print(Y)

'''
Missing data
'''
imputer = si(missing_values=nu.nan, strategy='mean')
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
#print(X)

'''
encoding
'''
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

cd = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],
remainder='passthrough')
X = nu.array(cd.fit_transform(X))
lb = LabelEncoder()
Y = lb.fit_transform(Y)
#print(X)
#print(Y)

'''
splitting
'''

from sklearn.model_selection import train_test_split

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y, test_size = 0.2, random_state = 1)

#print(Xtrain)
#print(Xtest)
#print(Ytrain)
#print(Ytest)

'''
feature scaling
'''

from sklearn.preprocessing import StandardScaler

sd = StandardScaler()
Xtrain[:, 3:5] = sd.fit_transform(Xtrain[:, 3:5])
Xtest[:, 3:] = sd.fit_transform(Xtest[:, 3:])
#print(Xtrain)

