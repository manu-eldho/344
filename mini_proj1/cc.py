import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tabulate import tabulate
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import svm, datasets

# import for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# import methods for measuring accuracy, precision, recall etc
from sklearn.metrics import (
    accuracy_score, 
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report)

import time

Data = pd.read_csv("C:\\Users\\manu eldho\\Downloads\\Datanew_feat_select (1).csv")
y = Data['class']

Data = Data.drop('class', axis=1)
X = Data

# stratified sampling to create a training and testing set
from sklearn.model_selection import StratifiedShuffleSplit

stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in stratified_split.split(X, y):
    X_train_strat, X_test_strat = X.iloc[train_index], X.iloc[test_index]
    y_train_strat, y_test_strat = y.iloc[train_index], y.iloc[test_index]
    
train_data_strat = pd.concat([X_train_strat, y_train_strat], axis=1)
test_data_strat = pd.concat([X_test_strat, y_test_strat], axis=1)

train_data_strat.to_csv('train_data_stratnew_feat_select.csv', index=False)
test_data_strat.to_csv('test_data_stratnew_feat_select.csv', index=False)

train_data = pd.read_csv('train_data_stratnew_feat_select.csv')
test_data = pd.read_csv('test_data_stratnew_feat_select.csv')

# View data 
print(f'Training data shape: {train_data_strat.shape}')
print(f'Testing data shape: {test_data_strat.shape}')
# print(f'Validation data shape: {valid_data_strat.shape}')

num_class_0 = train_data_strat['class'].value_counts()[0]
num_class_1 = train_data_strat['class'].value_counts()[1]
print(f"Number of samples in class 0: {num_class_0}")
print(f"Number of samples in class 1: {num_class_1}")

num_class_0 = test_data_strat['class'].value_counts()[0]
num_class_1 = test_data_strat['class'].value_counts()[1]

print(f"Number of samples in class 0: {num_class_0}")
print(f"Number of samples in class 1: {num_class_1}")

#Classification 
#Logistic Regression 
lr = LogisticRegression(C=2.5609999999999995,random_state=42)
lr.fit(X_train_strat, y_train_strat) 

#Weighted logistic regression 
w = {0: 1, 1: 1.5} 
lrw= LogisticRegression(random_state=42, class_weight=w, C=2.5609999999999995)
lrw.fit(X_train_strat, y_train_strat)

#Random Forest 
rf = RandomForestClassifier(n_estimators=200,max_depth=35,random_state=42)
rf.fit(X_train_strat, y_train_strat)

#Weighted Random Forest  
class_weights = {0: 1, 1: 2}
wrf = RandomForestClassifier(n_estimators=300,max_depth=47, class_weight=class_weights, random_state=2)
wrf.fit(X_train_strat, y_train_strat) 

#MLP 
hidden_layer_sizes = (10,) 
activation = 'relu'
mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation,random_state=42)
mlp.fit(X_train_strat, y_train_strat)

#XGBOOST
xgb= XGBClassifier(learning_rate= 0.3, max_depth=3, n_estimators= 150,random_state=42)
xgb.fit(X_train_strat, y_train_strat)

#SVM
rbf = svm.SVC(kernel='rbf', gamma=0.01, C=3)
rbf.fit(X_train_strat, y_train_strat) 

#Bootstap 
from sklearn.metrics import confusion_matrix

n_iterations = 1000
X_bs_list = []
y_bs_list = []

# classifiers predictions 
classifiers = [lrw, rf, wrf, mlp, xgb, rbf,lr]
classf = ['lrw', 'rf', 'wrf', 'mlp', 'xgb', 'rbf','lr']

accuracy = {}
precision = {}
recall = {}
fscore = {}
TNR_dict={}
TPR_dict={}

for clftext in classf :
    accuracy[clftext]=[]
    precision[clftext]=[]
    recall[clftext]=[]
    fscore[clftext]=[]
    TNR_dict[clftext]=[]
    TPR_dict[clftext]=[]

for i in range(n_iterations):
    x_testing=X_test_strat
    x_testing['y']=y_test_strat.values
    x_testing=x_testing.sample(n=40,replace=True)
    y_bs=x_testing['y']
    X_bs=x_testing
    X_bs.drop('y',axis=1,inplace=True)
    
    X_bs_list.append(X_bs)
    y_bs_list.append(y_bs)

    for clf,clftext in zip(classifiers,classf):

    # make predictions    
        y_hat = (clf.predict(X_bs))
    
    # print classification report
        report = classification_report(y_bs, y_hat, output_dict=True)
        
        p = round(report['weighted avg']['precision']*100,1)
        arr=precision[clftext]
        arr.append(p)
        precision[clftext]=arr
        
        r = round(report['weighted avg']['recall']*100,1)
        arr=recall[clftext]
        arr.append(r)
        recall[clftext]=arr
        
        f = round(report['weighted avg']['f1-score']*100,1)
        arr=fscore[clftext]
        arr.append(f)
        fscore[clftext]=arr
        
        score = round(report['accuracy']*100,1)
        arr=accuracy[clftext]
        arr.append(score)
        accuracy[clftext]=arr
        
        cm = confusion_matrix(y_bs, y_hat)
        TN, FP, FN, TP = cm.ravel()

                                              
        TNR = (TN / (TN + FP))*100 if (TN + FP) != 0 else 0
        arr=TNR_dict[clftext]
        arr.append(TNR)
        TNR_dict[clftext]=arr

def format_row(row, widths):
    return " | ".join(f"{item:{width}}" for item, width in zip(row, widths))

table_data = []
for i, clf,clftext in zip(range(0,len(classifiers)),classifiers,classf):
    avg_accuracy= sum(accuracy[clftext]) / len(accuracy[clftext])
    avg_precision = sum(precision[clftext]) / len(precision[clftext])
    avg_recall = sum(recall[clftext]) / len(recall[clftext])
    avg_fscore = sum(fscore[clftext]) / len(fscore[clftext])
    avg_tnr=sum(TNR_dict[clftext]) / len(TNR_dict[clftext])
    table_data.append([type(clf).__name__, f"{avg_accuracy:.4f}", f"{avg_precision:.4f}",f"{avg_recall:.4f}", f"{avg_tnr:.4f}", f"{avg_fscore:.4f}"])

column_widths = [max(len(str(row[i])) for row in table_data) for i in range(len(table_data[0]))]
header = ["Classifier", "Accuracy", "Precision", "Sensitivity","Specificity", " F1 score"]
header_widths = [max(len(header[i]), column_widths[i]) for i in range(len(header))]

print(format_row(header, header_widths))
print("-" * sum(header_widths) + "-" * (len(header_widths) - 1))
for row in table_data:
    print(format_row(row, header_widths))