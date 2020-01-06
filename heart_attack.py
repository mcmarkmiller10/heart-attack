# Use heart attack data and various algorithms to
# make predictions about heart disease levels in
# individuals in the test data.

import sys
import scipy
import numpy

import pandas
import sklearn

from pandas import read_csv

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn import datasets

filename = 'file://localhost/C:/Users/Mark/Documents/Test/processed.cleveland.csv'
attributes = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
dataset = read_csv(filename, names = attributes)

# account for missing data values
dataset[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']]=dataset[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']].replace('?', numpy.NaN)
dataset.dropna(inplace=True)

array = dataset.values
X = array[:,0:13]
y = array[:,13]
y=y.astype('int')
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.1, random_state=1)

# See how the linear regression model does not give sensible results!
model_reg = LinearRegression()
model_reg.fit(X_train,Y_train)
prediction_reg = model_reg.predict(X_validation)
print(prediction_reg)
result_reg = model_reg.score(X_validation,Y_validation)
print(result_reg)

model_lda = LinearDiscriminantAnalysis()
model_lda.fit(X_train,Y_train)
prediction_lda = model_lda.predict(X_validation)
print(prediction_lda)
result_lda = model_lda.score(X_validation, Y_validation)
print(result_lda)

model_svc = SVC(gamma='auto')
model_svc.fit(X_train,Y_train)
prediction_svc = model_svc.predict(X_validation)
print(prediction_svc)
result_svc = model_svc.score(X_validation, Y_validation)
print(result_svc)


a = input('press enter to continue...')
