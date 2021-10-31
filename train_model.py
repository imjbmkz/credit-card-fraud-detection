## Data Modeling
## Import packages
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
print('Packages have been loaded.')

## Get the training and testing data
train_data = pd.read_csv('train_data_cleaned.csv')
test_data = pd.read_csv('test_data_cleaned.csv')
print('Data has been loaded.')

## Split features and labels from training and testing sets
feature_names = train_data.filter(like='V').columns.to_list()
X_train = train_data[feature_names]
X_test = test_data[feature_names]
y_train = train_data.Class
y_test = test_data.Class
print('Features and labels have been separated.')

## Instantiate the models
logit = LogisticRegression()
tree = DecisionTreeClassifier()
svc = SVC()
forest = RandomForestClassifier(random_state=1)

## Fit data to the models
logit.fit(X_train, y_train)
tree.fit(X_train, y_train)
svc.fit(X_train, y_train)
forest.fit(X_train, y_train)
print('Models have been trained.')

## Make a list of models
models = [logit, tree, svc, forest]
model_names = [type(m).__name__ for m in models]

## Get ROC AUC scores on training data
roc_auc_scores = {}
for model in models:
  prediction = model.predict(X_train)
  roc_auc = round(roc_auc_score(y_train, prediction), 4)
  roc_auc_scores.update([(type(model).__name__, roc_auc)])
print('Training predictions were made. ROC AUC Scores generated.')
highest_roc_auc = max(roc_auc_scores, key=roc_auc_scores.get)
print('In training data, {} has the highest ROC AUC score of {}.'.format(highest_roc_auc, roc_auc_scores.get(highest_roc_auc)))

## Export training data with predictions 
train_data_predict = train_data.copy()
train_data_predict.loc[:, 'Predictions'] = model[model_names.index(highest_roc_auc)].predict(X_train)
train_data_predict.to_csv('train_data_predict.csv', header=True, index=False)
print('Training data with predictions has been exported to csv.')

## Get ROC AUC scores on testing data
roc_auc_scores = {}
for model in models:
  prediction = model.predict(X_test)
  roc_auc = round(roc_auc_score(y_test, prediction), 4)
  roc_auc_scores.update([(type(model).__name__, roc_auc)])
print('Testing predictions were made. ROC AUC Scores generated.')
highest_roc_auc = max(roc_auc_scores, key=roc_auc_scores.get)
print('In testing data, {} has the highest ROC AUC score of {}.'.format(highest_roc_auc, roc_auc_scores.get(highest_roc_auc)))

## Export testing data with predictions 
test_data_predict = test_data.copy()
test_data_predict.loc[:, 'Predictions'] = model[model_names.index(highest_roc_auc)].predict(X_test)
test_data_predict.to_csv('test_data_predict.csv', header=True, index=False)
print('Testing data with predictions has been exported to csv.')

print('All tasks have been completed.')