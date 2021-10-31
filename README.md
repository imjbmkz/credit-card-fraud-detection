# credit-card-fraud-detection
This is GitHub repository that contains the scripts used to predict fraud credit card transactions. 

## Data source
This [dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle was used in this task. It has 28 anonymized features and 2 that were kept as is (`Time` and `Amount`). It is highly imbalanced, having only **492** fraudulent transactions from **284807** records (**0.172%**). These are the credit card transactions by European cardholders recorded in September 2013.

## Data preprocessing 
The 28 anonymized features were the result of PCA transformation of the original features. There were no more processing conducted to these features as they have already been transformed. The `Time` and `Amount` variables, however, were standardized (<img src="https://render.githubusercontent.com/render/math?math=Z=\frac{x-\mu}{\sigma}">) using sklearn's `StandardScaler`. Finally, the data was splitted into training and testing sets (70% and 30% respectively) "_in a stratified fashion_"\[[1](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)].

## Feature selection
Due to the fact that most of the data were anonymized, features that were used to train the models were selected based from the **top 10 important features**. These were identified by training a random forest model using the training set. Then, using the `feature_importances_` property of class `RandomForestClassifier`, the feature importances of each variable were acquired and sorted to get the most important features of the model. After identifying the predictors to be used, both training and testing sets were filtered to remove the least necessary features from the data.

## Model training and evaluation
There were five (5) models that were used: logistic regression, decision trees, support vector classifier, random forest, and xgboost. Training data was passed on to these models with their default parameters. Then, the trained models were used to predict if the transaction is a fraudulent transaction or not. **ROC AUC** was used to measure the performance of the models since the data is highly imbalanced. **Decision tree** and **random forest** models had the highest ROC AUC scores respective to the training and testing sets.

## Scripts used
To reproduce the results, download the data from this [link](https://www.kaggle.com/mlg-ulb/creditcardfraud). Ensure that the current working directory has the `creditcard.csv` file to proceed. Using your console, run the `split_and_clean_data.py` and wait for the message "All tasks have been completed". You should also notice that `train_data_cleaned.csv` and `test_data_cleaned.csv` were extracted to your working directory. Then, run the `train_model.py` and wait for the message "All tasks have been completed". This script will give you `train_data_predict.csv` and `test_data_predict.csv` which is same data with predictions made by the 'best' model/s.
