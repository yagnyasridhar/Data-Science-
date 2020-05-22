import pandas as pd 
import matplotlib.pyplot as plt
import numpy as nu
import seaborn as sb 

sb.set()
class preprocessor:
    __path = ""
    x = None
    y = None
    data = None

    def __init__(self, path):
        self.__path  = path
    
    def describe():
        print(data.describe())

    def feed(self, i, j):
        self.data = pd.read_csv(self.__path)
        self.x = self.data.iloc[:, i].values
        self.y = self.data.iloc[:, j].values
        return self.x, self.y
    
    def missingData(self,X):
        from sklearn.impute import SimpleImputer
        imputerObj = SimpleImputer(missing_values = 0, strategy="mean")
        #print(X)
        imputerObj = imputerObj.fit(X)
        return imputerObj.transform(X)
    
    def lblencoder(self,X):
        from sklearn.preprocessing import LabelEncoder
        lbl = LabelEncoder()
        return lbl.fit_transform(X)
    
    def hotEncoder(self, X, data):
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder(categories= 0)
        return enc.fit_transform(X).toarray()

    def split(self, x,y):
        from sklearn.model_selection import train_test_split
        return train_test_split(x, y, test_size = 1/3, random_state = 123)
    
    def scale(self, xtrain,xtest):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        xtrain = sc.fit_transform(xtrain)
        xtest = sc.transform(xtest)
        return xtrain, xtest

# Preprocessing 
obj = preprocessor("Salary_Data.csv")
x,y = obj.feed(0,1)
xtrain,xtest,ytrain,ytest = obj.split(x,y)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain.reshape(-1,1), ytrain.reshape(-1,1))

ypred = regressor.predict(xtest.reshape(-1,1))

plt.scatter(xtrain, ytrain, Color = "Red")
plt.plot(xtrain, regressor.predict(xtrain.reshape(-1,1)),color="Blue")
plt.title("Salary vs Experience")
plt.show()

plt.scatter(xtest, ytest, Color = "Red")
plt.plot(xtrain, regressor.predict(xtrain.reshape(-1,1)), color="Blue")
plt.title("Salary vs Experience")
plt.show()

print(regressor.score(x.reshape(-1,1), y.reshape(-1,1)))