""" Random forest solution """

# importing necessary files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def catego_nan(data,s):
    """ Fills up nan in categorical variables """
    data = data.iloc[:,:]

    data[s] = data[s].fillna('no')
    return data


def numero_nan(data,s):
    """ Fills up nan in numerical variables """
    data = data.iloc[:,:]

    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(data[s].values.reshape(-1,1))
    data[s] = imputer.transform(data[s].values.reshape(-1,1))
    return data


def encode(data,s):
    """ Encodes binary data into ones and zeros """
    data = data.iloc[:,:]

    LE = LabelEncoder()
    data[s] = LE.fit_transform(data[s])
    return data


# data preprocessing steps
def data_preprocess(data):
    ''' Data preprocessing '''
    data = data.iloc[:,:]

    categorical_var = ['International_Plan','Voice_Mail_Plan']
    numerical_var = [i for i in data.columns if i not in categorical_var]

    # first step is replacing nans
    # replacing categorical nans with 'no'
    for s in categorical_var:
        data = catego_nan(data,s)

    # replacing numerical nans
    for s in numerical_var:
        data = numero_nan(data,s)


    # encoding of categorical variables
    for s in categorical_var:
        data = encode(data,s)

    return data


# importing tarining dataset
dataTrain = pd.read_csv('churnTrain.csv')

# removing unwanted data
dataTrain = dataTrain.drop(columns=['State','Area_Code','Phone_No'])

# data preprocessing, mainly replacing nan's and encoding

# dividing into X and y
X_train = dataTrain.iloc[:,:-1]
y_train = dataTrain.iloc[:,-1]

# nan's and encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer

X_train = data_preprocess(X_train)
y_train = encode(y_train.to_frame('Churn'),'Churn')


# converting the dataFrames to arrays
X_train = X_train.values
y_train = y_train.values


# building a simple classification model based on random forest
from sklearn.ensemble import RandomForestClassifier
Classifier = RandomForestClassifier(n_estimators=500)
Classifier.fit(X_train,y_train.ravel())


# importing testing set
dataTest = pd.read_csv('churnTest.csv')

# removing unwanted data
dataTest = dataTest.drop(columns=['State','Area_Code','Phone_No'])

# getting X_test
X_test = dataTest.iloc[:,:]
X_test = data_preprocess(X_test)
X_test = X_test.values


# predicting y for X_test
y_pred = Classifier.predict(X_test)

# saving as csv file
y_pred = np.append(np.array(range(1,426)).reshape(-1,1),y_pred.reshape(-1,1),axis=1)
y_pred = pd.DataFrame(y_pred,columns=['Id','Churn'])

for i in range(425):
    if y_pred.iloc[i,1] == 0:
        y_pred.iloc[i,1] = "FALSE"
    else:
        y_pred.iloc[i,1] = "TRUE"


y_pred.to_csv('y_pred.csv',sep=',',index=False)
