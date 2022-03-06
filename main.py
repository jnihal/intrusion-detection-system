'''
Link to the dataset: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
'''

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from time import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python file_name.py data_directory [small]")
    
    # Check if the user opted for smaller dataset
    if len(sys.argv) == 3:
        small = True
    else:
        small = False

    # Load the data
    print('Loading data...')
    data_frame = load_data(sys.argv[1], small)

    # Preprocessing the data
    print('Preprocessing...')
    X, Y = preprocessing(data_frame)

    # Split test and train data 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Perform machine learning
    print("\nDecision Tree")
    model = DecisionTreeClassifier(criterion="entropy", max_depth=4)
    result(model, X_train, X_test, Y_train, Y_test)

    print("\nRandom Forest")
    model = RandomForestClassifier(n_estimators=30)
    result(model, X_train, X_test, Y_train, Y_test)
  

# Load the data into memory using pandas
def load_data(directory, small):

    # Create a list of all the features
    features = []
    path = os.path.join(directory, 'kddcup.names.txt')
    with open(path, 'r') as f:
        next(f)
        for line in f:
            features.append(line.split(':')[0])
        
    features.append('target')

    # Create a dictionary of all the attack types
    attacks = {}
    path = os.path.join(directory, 'kddcup.attack_types.txt')
    with open(path, 'r') as f:
        for line in f:
            words = line.split()
            attacks[words[0]] = words[1]

    attacks['normal'] = 'normal'

    # Create a dataframe using the appropriate dataset(small or large)
    if small:
        path = os.path.join(directory, 'kddcup.data_10_percent.csv')
    else:
        path = os.path.join(directory, 'kddcup.data.csv')

    df = pd.read_csv(path, names=features)

    df['target'] = df.target.apply(lambda cell: attacks[cell[:-1]])

    # Return the dataframe
    return df


# Prepares the raw data for machine learning
def preprocessing(df):

    # Remove rows that have null values
    df = df.dropna()

    # Remove columns that have the same value for every record
    to_drop = [col for col in df.columns if df[col].nunique() == 1]

    # Feature selection using Pearson's Coeffecient to remove highly correlated features
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop.extend([column for column in upper.columns if any(upper[column] > 0.95)])
    to_drop.extend(['service'])
    df.drop(to_drop, axis=1, inplace=True)

    # Convert categorical features into numerical form for better efficiency
    num_cols = df._get_numeric_data().columns
    cate_cols = list(set(df.columns) - set(num_cols))
    cate_cols.remove('target')

    for col in cate_cols:
        label = LabelEncoder()
        label.fit(df[col])
        df[col] = label.transform(df[col])

    # Split the data into independent(X) and dependent(Y) variables
    Y = df[['target']]
    X = df.drop(['target', ], axis=1)

    # Scale down the X values between 0 to 1
    sc = MinMaxScaler()
    X = sc.fit_transform(X)

    # Return the variables
    return X, Y


def result(model, X_train, X_test, Y_train, Y_test):

    # Train the model
    print('Training...')
    start = time()
    model.fit(X_train, Y_train.values.ravel())
    end = time()
    print("Training time: ", end - start)

    # Predict the outputs
    start = time()
    Y_test_prediction = model.predict(X_test)
    end = time()
    print("Testing time: ", end - start)

    # Print the results
    print("Accuracy: ", accuracy_score(Y_test, Y_test_prediction))
    print("Report:\n", classification_report(Y_test, Y_test_prediction, zero_division=1, digits=4))

    # Construct a confusion matrix
    plot_confusion_matrix(model, X_test, Y_test, normalize='true', cmap='Blues', values_format='.4f')

    # Save the confusion matrix
    plt.savefig(str(model))


if __name__ == "__main__":
    main()