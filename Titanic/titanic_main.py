import pandas as pd
import os

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import prepare_data as etl

from functions import evaluate_model, get_feature_predictivity, create_evaluation_plots

pd.options.display.max_columns = None

root = os.getcwd()

# Load data

train = pd.read_csv(root + r'\Data\train.csv')
test = pd.read_csv(root + r'\Data\test.csv')

train['dataset'] = 'train'
test['dataset'] = 'test'

raw_data_combined = pd.concat([train, test], ignore_index=True)

# Call function to create features and final training dataset
combine = etl.titanic_process_data(raw_data_combined)

# Drop dataset recognition columns
train_df = combine[combine['dataset'] == 'train'].drop(columns=['dataset'])
test_df = combine[combine['dataset'] == 'test'].drop(columns=['dataset'])

# Split the training data into training and out-of-sample (OOS) test sets
X_train, X_test_oos, y_train, y_test_oos = train_test_split(train_df.drop('Survived', axis=1),
                                                            train_df['Survived'],
                                                            test_size=0.3,
                                                            random_state=42)

# 'Survived' is not in the original test dataset; it is created during the concatenation with pd.concat and is filled with NaNs.
X_test = test_df.drop('Survived', axis=1)

# Initiate and fit the model
model = SVC(verbose=True, probability=True)
model.fit(X_train, y_train)

# Predicting Out-of-sample
y_pred_oos = model.predict(X_test_oos)

# Predicting on unseen dataset
y_pred = model.predict(X_test)


# Evaluate model OOS and save metrics to folder
evaluate_model(model, X_test_oos, X_test_oos, y_test_oos, y_pred_oos)
get_feature_predictivity(model, X_train, y_train)
create_evaluation_plots(model, X_train, X_test_oos, y_train, y_test_oos)


# Export csv for submission to Kaggle
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred.astype(int)
    })

submission.to_csv('../Titanic/submission.csv', index=False)
