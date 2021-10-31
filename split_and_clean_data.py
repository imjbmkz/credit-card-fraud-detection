## Data Cleaning
## Import packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
print('Packages have been loaded.')

## Get the raw data
credit_card = pd.read_csv('creditcard.csv')
print('Data has been loaded.')

## Get feature names
feature_names = credit_card.loc[:,:'Amount'].columns.to_list()

## Separate predictor and response variables
X = credit_card[feature_names].to_numpy()
y = credit_card.Class.to_numpy()

## Split training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=1, test_size=0.3)

## Define preprocessor
standardize = StandardScaler()

## Standardize Time variable
X_train[:, 0] = standardize.fit_transform(X_train[:, 0].reshape(-1 ,1)).reshape(1,-1)
X_test[:, 0] = standardize.transform(X_test[:, 0].reshape(-1, 1)).reshape(1, -1)

## Standardize Amount variable
X_train[:, 29] = standardize.fit_transform(X_train[:, 29].reshape(-1 ,1)).reshape(1,-1)
X_test[:, 29] = standardize.transform(X_test[:, 29].reshape(-1, 1)).reshape(1, -1)

print('Data has been transformed.')

## Get feature importance
forest = RandomForestClassifier(random_state=1, max_depth=5)
forest.fit(X_train, y_train)
print('Model has been trained.')

forest_importances = pd.Series(forest.feature_importances_, index=feature_names)
forest_importances.sort_values(ascending=False, inplace=True)
important_features_10 = forest_importances.head(10).index
print('Features have been selected.')

features_to_keep = []
for feature in important_features_10:
  features_to_keep.append(feature_names.index(feature))

## Select the data that will be needed based from the features identified
X_train_select = X_train[:, features_to_keep]
X_test_select = X_test[:, features_to_keep]

## Append X with the y values
X_train_data = np.append(X_train_select, y_train.reshape(-1, 1), axis=1)
X_test_data = np.append(X_test_select, y_test.reshape(-1, 1), axis=1)

features_to_keep_2 = list(feature_names[i] for i in features_to_keep)
features_to_keep_2.append('Class')

## Create dataframes
train_data = pd.DataFrame(X_train_data, columns=features_to_keep_2)
test_data = pd.DataFrame(X_test_data, columns=features_to_keep_2)

## Import data frames to csv
train_data.to_csv('train_data_cleaned.csv', header=True, index=False)
test_data.to_csv('test_data_cleaned.csv', header=True, index=False)
print('Data for training and testing models has been exported to csv.')
print('All tasks have been completed.')