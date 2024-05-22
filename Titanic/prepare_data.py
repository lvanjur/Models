import numpy as np
import pandas as pd
import re


def titanic_process_data(df):
    
    # Extract title
    df['Title'] = df['Name'].apply(lambda row:
                                   re.search(r' ([A-Za-z]+)\.', row).group().strip())

    # Extract last name
    df['Family_name'] = df['Name'].apply(lambda row:
                                         re.search(r'([A-Za-z]+)\,', row).group().rstrip(','))

    # Create Age_group feature
    df['Age_estimated'] = df[['Title', 'Age']].groupby('Title').transform(
        lambda x: x.fillna(x.mean()))
    df['Age_estimated'] = round(df['Age_estimated'], 1)

    df['Age_group'] = np.where(df['Age_estimated'] <= 14, 1,
                               np.where(df['Age_estimated'] <= 67, 2,
                                        np.where(df['Age_estimated'] > 67, 3, -999)))

    # Create Fare_group feature
    df['Fare'] = df[['Pclass', 'Fare']].groupby('Pclass').transform(
        lambda x: x.fillna(x.mean()))

    df['Fare_group'] = pd.qcut(df['Fare'], 3)
    df['Fare_group'] = np.where(df['Fare'] <= 8.7, 1,
                                np.where(df['Fare'] <= 26, 2,
                                         np.where(df['Fare'] > 26, 3, -999)))

    df['Class'] = df['Fare_group'] / df['Pclass']
    df['Class_binary'] = np.where(df['Class'] <= 1, 1, 2)

    columns_to_encode = ['Sex', 'Age_group', 'Fare_group', 'Class_binary']

    for column in columns_to_encode:
        df = df.join(pd.get_dummies(df[column], prefix=f'{column}').astype(int))

    features_to_drop = ['PassengerId', 'Name', 'Age', 'Age_group',
                        'Ticket', 'Cabin', 'Fare_group',
                        'Embarked', 'Title', 'Family_name',
                        'Fare', 'Age_estimated', 'Sex', 'Class_binary',
                        'Pclass', 'Class', 'SibSp', 'Parch']

    df = df.drop(features_to_drop, axis=1)

    # Dropping redundant columns
    df = df.drop(['Sex_male', 'Class_binary_2', 'Fare_group_1', 'Fare_group_2',
                  'Age_group_2', 'Age_group_3'], axis=1)
    
    df.rename(columns={
        'Sex_female': 'Female_YN',
        'Age_group_1': 'Younger_than_14_YN',
        'Fare_group_3': 'Expensive_ticket_YN',
        'Class_binary_1': 'Low_class_YN'
        }, inplace=True)

    return df
