# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 21:22:57 2020

@author: AbdelAzeem
"""

import warnings
warnings.filterwarnings('ignore') 
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
#----------------------------------------------------
#reading data

data = pd.read_csv('heart.csv')
#data.describe()

#X Data
X = data.iloc[:,:-1]
#y Data
y = data.iloc[:,-1]
print('X Data is \n' , X.head())
print('X shape is ' , X.shape)


# -------------- MinMaxScaler for Data --------------
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)


#---------Feature Selection = Logistic Regression 13=>7 -------------------

from sklearn.linear_model import  LogisticRegression

thismodel = LogisticRegression()


FeatureSelection = SelectFromModel(estimator = thismodel, max_features = None) # make sure that thismodel is well-defined
X = FeatureSelection.fit_transform(X, y)

#showing X Dimension 
print('X Shape is ' , X.shape)
print('Selected Features are : ' , FeatureSelection.get_support())


#------------ Splitting data ---33% Test  67% Training -----------------------

#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)

#Splitted Data
print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)
#-----------------------------------------------------

#    ------- AdaBoost classifier ----- 84 % --- n_estimators=25, 27-------

from sklearn.ensemble import AdaBoostClassifier


'''
sklearn.ensemble.AdaBoostClassifier(base_estimator=None, n_estimators=50,
                                    learning_rate=1.0, algorithm=’SAMME.R’,
                                    random_state=None)
'''

# ----- 84 % @ 25,27 -----

model = AdaBoostClassifier(n_estimators=50)
import time
t0 = time.clock()
model.fit(X_train,y_train)
tr = (time.clock()-t0)
print('AdaBoost Classifier Model Train Score is : ' , model.score(X_train, y_train))
print('AdaBoost Classifier Model Test Score is : ' , model.score(X_test, y_test))
print('-------------------------')
print('time in msec = ', tr*1000)
print('-------------------------')
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)

# ------------------ Metrics ----------------------
# ---------- confusion_matrix ----------

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)



from sklearn.metrics import roc_auc_score
#5 roc_auc_score(y_true, y_score, average=’macro’, sample_weight=None,max_fpr=None)
TP = 48
TN = 37
FN = 6
FP = 9
accuracy_score = ((TP + TN) / float(TP + TN + FP + FN))*100
precision_score = (TP /float(TP + FP ))*100
recall_score = (TP / float(TP + FN))*100
f1_score = (2 * (precision_score * recall_score) / (precision_score + recall_score))
print('accuracy_score is :' , accuracy_score)
print('Precision Score is : ', precision_score)
print('recall_score is : ', recall_score)
print('f1_score is :' , f1_score)
ROCAUCScore = roc_auc_score(y_test,y_pred, average='micro') #it can be : macro,weighted,samples
print('ROCAUC Score : ', ROCAUCScore*100)

# --------------------------------------------------

########################### ((Grid Search)) ##############

from sklearn.ensemble import AdaBoostClassifier


from sklearn.model_selection import GridSearchCV
import pandas as pd

SelectedModel = AdaBoostClassifier()

'''
sklearn.ensemble.AdaBoostClassifier(base_estimator=None, n_estimators=50,
                                    learning_rate=1.0, algorithm=’SAMME.R’,
                                    random_state=None)
'''

SelectedParameters = {'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.6,0.6,0.7,0.8,0.9,1,2,3,4,5,10],
                      'n_estimators':[10,30,50,70,80,90,100]}
t0 = time.clock()
GridSearchModel = GridSearchCV(SelectedModel,SelectedParameters, cv = 10,return_train_score=True)

GridSearchModel.fit(X_train, y_train)
tr_time = (time.clock() -t0)
sorted(GridSearchModel.cv_results_.keys())

GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]

# Showing Results
print('All Results are :\n', GridSearchResults )
print('Best Score is :', GridSearchModel.best_score_)
print('Best Parameters are :', GridSearchModel.best_params_)
print('Best Estimator is :', GridSearchModel.best_estimator_)
print('Train time in Sec = : ' , tr_time*1000)

# ========= optimized results ==========
'''
Best Score is : 0.8419047619047619
Best Parameters are : {'learning_rate': 0.1, 'n_estimators': 30}
Best Estimator is : AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=0.1,
                   n_estimators=30, random_state=None)
Train time in Sec = :  133913.93659999996
'''
# ========= optimized results ==========

model = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
import time
t0 = time.clock()
model.fit(X_train,y_train)
tr = (time.clock()-t0)
print('AdaBoost Classifier Model Train Score is : ' , model.score(X_train, y_train))
print('AdaBoost Classifier Model Test Score is : ' , model.score(X_test, y_test))
print('-------------------------')
print('time in msec = ', tr*1000)
print('-------------------------')
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)

# ------------------ Metrics ----------------------
# ---------- confusion_matrix ----------

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)



from sklearn.metrics import roc_auc_score
#5 roc_auc_score(y_true, y_score, average=’macro’, sample_weight=None,max_fpr=None)
TP = 48
TN = 37
FN = 6
FP = 9
accuracy_score = ((TP + TN) / float(TP + TN + FP + FN))*100
precision_score = (TP /float(TP + FP ))*100
recall_score = (TP / float(TP + FN))*100
f1_score = (2 * (precision_score * recall_score) / (precision_score + recall_score))
print('accuracy_score is :' , accuracy_score)
print('Precision Score is : ', precision_score)
print('recall_score is : ', recall_score)
print('f1_score is :' , f1_score)
ROCAUCScore = roc_auc_score(y_test,y_pred, average='micro') #it can be : macro,weighted,samples
print('ROCAUC Score : ', ROCAUCScore*100)

# --------------------------------------------------
