import pandas as pd
import numpy as np
from sklearn import preprocessing
from math import ceil
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import pickle
from sklearn.externals import joblib


SVM = joblib.load('SVM.pkl')                        #load the models
RF = joblib.load('RandomForest.pkl') 

trainingData = pd.read_csv('TrainingDataPD.csv')    #load the files   
testData = pd.read_csv('testSold.csv')
groundTruth = pd.read_csv('gt.csv')
testData_copy = testData
cols = [1]
trainingData.drop(trainingData.columns[cols],axis=1,inplace=True)            #dropping the id value column 
groundTruth = groundTruth.drop(columns=['Id'])       
testData = testData.drop('Id', axis = 1)

#preprocessing steps
cols_to_transform = ['MSZoning','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','MasVnrType','GarageType','GarageFinish','SaleCondition' ]                     #columns with texual information

testData[['MasVnrArea','GarageYrBlt','BsmtFullBath','BsmtHalfBath','GarageCars', 'GarageArea']] = testData[['MasVnrArea','GarageYrBlt','BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea']].fillna(value=0) #cleanup the data
testData[['MasVnrType']] = testData[['MasVnrType']].fillna(value='None')


testData = pd.get_dummies(testData, columns = cols_to_transform)            #encoding categorical values

cleanup_nums = {"SaleStatus": {"SoldFast": 1, "SoldSlow": 2,'NotSold':3}}
groundTruth = groundTruth.replace(cleanup_nums)

missing_cols = set( trainingData.columns ) - set( testData.columns )        #making sure no of columns in test and train are same
for c in missing_cols:
    testData[c] = 0

testData = testData.reindex_axis(sorted(testData.columns), axis=1)
testData = testData[trainingData.columns]

testData = testData.as_matrix(columns=None) 
groundTruth = groundTruth.as_matrix(columns=None)
testData = preprocessing.scale(testData)

predicted_values = RF.predict(testData)
predicted_value_text = []

for i in predicted_values:
    if(i == 1):
        predicted_value_text.append('SoldFast')
    elif(i == 2):
        predicted_value_text.append('SoldSlow')
    
    elif(i == 3):
        predicted_value_text.append('NotSold')

predicted_df = pd.DataFrame({'col':predicted_value_text})  
id_column = testData_copy[['Id']]

out_df = pd.concat([id_column, predicted_df], axis=1)
out_df.to_csv('out.csv',index=False)

print("Accuracy with SVM", SVM.score(testData, groundTruth))
print("Accuracy with RF", RF.score(testData, groundTruth))



    
