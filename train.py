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
from sklearn.externals import joblib

path = 'trainSold.csv'
df = pd.read_csv(path)
df = df.drop('Id', axis = 1)

cols_to_transform = ['MSZoning','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','MasVnrType','GarageType','GarageFinish','SaleCondition' ]                     #columns with texual information


df[['MasVnrArea','GarageYrBlt']] = df[['MasVnrArea','GarageYrBlt']].fillna(value=0)
df[['MasVnrType']] = df[['MasVnrType']].fillna(value='None')


df_dummy = pd.get_dummies(df, columns = cols_to_transform)                 #converting the textual data to numeric one hot encoding

cleanup_nums = {"SaleStatus": {"SoldFast": 1, "SoldSlow": 2,'NotSold':3}}
df_dummy = df_dummy.replace(cleanup_nums)

df_dummy = df_dummy.sample(frac=1)          #shuffling the rows
ylabel = df_dummy['SaleStatus']
df_dummy = df_dummy.drop(columns=['SaleStatus'])                           #removing the 'y' column from the original data


df_dummy = df_dummy.reindex_axis(sorted(df_dummy.columns), axis=1)
df_dummy.to_csv('TrainingDataPD.csv')

data_X = df_dummy.as_matrix(columns=None)                                  #converting to numpy arrays
data_Y = ylabel.as_matrix(columns=None)

data_X = preprocessing.scale(data_X)                                       #standardizing the data_X

total_rows = int(data_X.shape[0])
no_train_samples = int(ceil(0.6*total_rows))
no_cv_samples = int(ceil(0.2*total_rows))
no_test_samples = no_cv_samples
temp1 = no_train_samples+1
temp2 = no_train_samples+no_cv_samples
temp3 = temp2 + 1

train_X = data_X[0:no_train_samples,:]                  #dividing the data                                 
train_Y = data_Y[0:no_train_samples]

cv_X = data_X[temp1:temp2,:]
cv_Y = data_Y[temp1:temp2]

test_X = data_X[temp3:,:]
test_Y = data_Y[temp3:] 

#Train different models now

#logistic regression

C_array = np.linspace(0.1, 4, num = 25)
lr_train_acc = []
lr_cross_validation_acc = []

for c_temp in C_array:                                                  #hyperparameter tuning
    lr = LogisticRegression(C = c_temp, solver = 'sag',max_iter = 500)
    lr.fit(train_X,train_Y)
    lr_train_acc.append(lr.score(train_X, train_Y))
    lr_cross_validation_acc.append(lr.score(cv_X, cv_Y))


lr_chosen_c = C_array[lr_cross_validation_acc.index(max(lr_cross_validation_acc))]
lr = LogisticRegression(C = lr_chosen_c, solver = 'sag', max_iter=500)
lr.fit(train_X, train_Y)

joblib.dump(lr, 'LogisticRegression.pkl')

print("Test Accuracy with the chosen value of c for LogisticRegression is", lr.score(test_X, test_Y))

plt.figure(1)
plt.plot(C_array, lr_train_acc)
plt.xlabel("value of C")
plt.ylabel("training accuracy")
plt.title("Training Accuracy vs the hyperparameter C for LR")


plt.figure(2)
plt.plot(C_array, lr_cross_validation_acc)
plt.xlabel("value of C")
plt.ylabel("cross_validation accuracy")
plt.title("Cross Validation Accuracy vs the hyperparameter C for LR")



#Support Vector Machine

C_array_exp = np.linspace(-5, 10, num = 20)
gamma_array_exp = np.linspace(-15, 3, num = 20)

C_array = [2**i for i in C_array_exp]
gamma_array = [2**i for i in gamma_array_exp]

temp_array = []
for i in C_array:                       #hyperparameter tuning
    for j in gamma_array:               
        temp_array.append([i,j])

svm_train_acc = []
svm_cross_validation_acc = []

for hype in temp_array:
   
    svm_clf = svm.SVC(C = hype[0], gamma = hype[1] )
    svm_clf.fit(train_X,train_Y)
    svm_train_acc.append(svm_clf.score(train_X, train_Y))
    svm_cross_validation_acc.append(svm_clf.score(cv_X, cv_Y))

chosen_c = temp_array[svm_cross_validation_acc.index(max(svm_cross_validation_acc))][0]
chosen_gamma = temp_array[svm_cross_validation_acc.index(max(svm_cross_validation_acc))][1]
svm_clf = svm.SVC(C = chosen_c, gamma = chosen_gamma)
svm_clf.fit(train_X, train_Y)

joblib.dump(svm_clf, 'SVM.pkl')

print("Test Accuracy with the chosen value of hyperparameters for SVM is", svm_clf.score(test_X, test_Y))


##Random Forest

n_estimators_array = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_depth_array = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth_array.append(None)
min_samples_split_array = [ 2, 3, 4, 5, 6, 7, 8, 9, 10]
min_samples_leaf_array =  [1, 2, 4, 10, 15, 20, 50, 80, 100]

 #'max_features': max_features_array,
random_grid = {'n_estimators': n_estimators_array,
              
               'max_depth': max_depth_array,
               'min_samples_split': min_samples_split_array,
               'min_samples_leaf': min_samples_leaf_array,
              }

rf_clf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf_clf, param_distributions = random_grid, n_iter = 500, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(train_X, train_Y)

joblib.dump(rf_random, 'RandomForest.pkl')

print("Train accuracy is", rf_random.score(train_X, train_Y))
print("Test Accuracy with the chosen value of hyperparameters for RF is", rf_random.score(test_X, test_Y))


plt.show()














