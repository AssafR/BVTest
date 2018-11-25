import os
import pickle
import time

import numpy as np
import pandas as pd
import re as re

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

years_on_file_bands = ["0", "0.5", "1", "1.5", "2", "2.5", "3", "5", "6", "8", "10", "12", "14", "17", "20", "25",
                       "30", "35", "50"]
columns_for_years_bands = ["years_file_upto_" + x for x in years_on_file_bands]
regex_years = re.compile('between (.*?) and (.*?) years')

frame_statistic_functions = [  # Array of different statistics report for a frame
    ["Head", lambda frame: frame.head()],
    ["Data Description", lambda frame: frame.describe()],
    ["Data Types", lambda frame: frame.info()],
    ["Number of NaN values", lambda frame: frame.isna().sum()],
    ["Number of unique values", lambda frame: frame.nunique()]
]

column_statistic_functions = [
    ["value counts", lambda frame, column: frame[column].value_counts()]
]

categorical_columns_functions = [
    ["unique values", lambda frame, column: frame[column].unique()]
]


def convert_years_to_boolean(source_row):
    years_description = source_row['years_on_file']
    match = regex_years.search(years_description)
    max_index = years_on_file_bands.index(match.group(2)) + 1
    new_array = [0] * len(years_on_file_bands)
    new_array[0:max_index] = [1] * max_index
    row = dict(zip(columns_for_years_bands, new_array))  # Return a dictionary, column:value
    return row


def train_and_predict(df, dropped_columns):
    # For clients, only use the latest known data point
    sorted = df.sort_values(by='time')
    df = sorted.drop_duplicates(['client_id'], keep='last')

    df = df.drop(dropped_columns, axis=1)
    df = df.astype(float)

    # Extract the target variable
    y = df.pop('tag_in_six_months').values
    X = df

    # Split to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, shuffle=True)

    # Scaling
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    predictions = train_predict_logistic(X_test_scaled, X_train_scaled, y_train)
    print("Results for Logistic Regression:")
    print(classification_report(y_test, predictions))

    predictions = train_predict_XGB(X_test_scaled, X_train_scaled, y_train)
    print("Results for GradientBoostingClassifier:")
    print(classification_report(y_test, predictions))


def train_predict_XGB(X_test_scaled, X_train_scaled, y_train):
    # Train
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1, subsample=0.5)
    clf.fit(X_train_scaled, y_train)
    # Predict
    predictions = clf.predict(X_test_scaled)
    return predictions


def train_predict_logistic(X_test_scaled, X_train_scaled, y_train):
    # Train
    logisticRegr = LogisticRegression(solver='lbfgs')
    logisticRegr.fit(X_train_scaled, y_train)
    # Predict
    predictions = logisticRegr.predict(X_test_scaled)
    return predictions


def add_years_boolean(df):
    """Create the new columns based on 'years_on_file' category """

    # New dataframe with same indices but new columns
    df2 = pd.DataFrame(index=df.index,
                       columns=columns_for_years_bands)  # New dataframe, same index new columns

    filename = "./df2.pkl"  # Cache
    if os.path.isfile(filename):
        df2 = pickle.load(open(filename, "rb"))
    else:
        for (idx, row) in df.iterrows():
            boolean_columns_dict = convert_years_to_boolean(row)
            df2.iloc[idx] = boolean_columns_dict
        pickle.dump(df2, open(filename, "wb"))  # Cache
    df = df.join(df2)  # Join to add new columns
    return df


def create_Nan_indicator_column(dataframe, column, new_column_name, impute, imputed_value):
    """Create new column, populate 1/0 for missing/exists"""
    dataframe[new_column_name] = dataframe[column].isnull().map({True: 1, False: 0})
    if impute:
        dataframe[column].fillna(value=imputed_value, inplace=True)
    return dataframe

def run_tests_on_data(df):

    dropped_columns = ['years_file_upto_0', 'client_id', 'pit', 'years_on_file']
    all_columns_except_current_model = set(df.columns.values) - set(['Current_model_probability', 'tag_in_six_months'])

    client_data_first_funded_120 = df.loc[df['period'] == 0]
    client_data_first_funded_180 = df.loc[df['period'] == 1]
    print("\r\n------------\r\nBaseline - Current model probability\r\n")
    print("Model for first_funded+120:\r\n------------------------")
    train_and_predict(client_data_first_funded_120, all_columns_except_current_model)
    print("Model for first_funded+180:\r\n------------------------")
    train_and_predict(client_data_first_funded_180, all_columns_except_current_model)

    print("\r\n------------\r\nFull data - model\r\n")
    print("Model for first_funded+120:\r\n------------------------")
    train_and_predict(client_data_first_funded_120, dropped_columns)
    print("Model for first_funded+180:\r\n------------------------")
    train_and_predict(client_data_first_funded_180, dropped_columns)

pd.options.display.max_columns = 999


def main():
    client_data = pd.read_csv("./data_science_exam_bluevine_file.csv")

    print('Some Data exploration')
    for func in frame_statistic_functions:
        print("\r\n%s:\r\n%s" % (func[0], func[1](client_data)))
    for func in column_statistic_functions:
        print("\r\n%s:\r\n%s" % (func[0], func[1](client_data, 'period')))
    for column in ['period', 'segment', 'years_on_file', 'missing_report', 'client_industry_unknown',
                   'tag_in_six_months']:
        for func in categorical_columns_functions:
            print("%s for column %s:\r\n%s\r\n" % (func[0], column, func[1](client_data, column)))


    # Handle the special time field
    client_data['time'] = (pd.to_datetime(client_data['pit']))
    minimum_date = client_data['time'].min()
    client_data['time'] = ((client_data['time'] - minimum_date) / np.timedelta64(1, 'D')).astype(int)  # Count in days

    #Fix and impute data
    nan_indicator_new_columns = [
        ("sum_failed_repayments", "missing_sum_failed_repayments", True, 0),
        ("years_on_file", "missing_years_on_file", True, "between 0 and 0 years"),
        ("recent_successful_repayments", "missing_recent_successful_repayments", True, 0),
        ("balance", "missing_balance", True, 0),
        ("max_failed_repayments", "missing_max_failed_repayments", True, 0),
        ("credit_score", "missing_credit_score", True, 0),
        ("credit_inquiries_count", "missing_credit_inquiries_count", True, 0),
        ("credit_open_balance", "missing_credit_open_balance", True, 0),
        ("max_successful_repayments", "missing_max_successful_repayments", True, 0),
        ("available_credit", "missing_available_credit", True, 0),
    ]

    # New column indicating there is no data in this important field + imputation if necessary
    for fill_info in nan_indicator_new_columns:
        client_data = create_Nan_indicator_column(client_data, fill_info[0], fill_info[1], fill_info[2], fill_info[3])
    categorical_replacements = \
        {"period": {"first_funded+120": 0, "first_funded+180": 1},
         "segment": {"old": 0, "new": 1},
         "tag_in_six_months": {"good": 0, "bad": 1},
         "years_on_file": {"less than 0.5 year": "between 0 and 0.5 years",
                           "more than 35 years": "between 35 and 50 years",
                           },
         }

    client_data.replace(categorical_replacements, inplace=True)
    print("\r\nAfter categorical encoding\r\n------------------------")
    for column in ['period', 'segment', 'years_on_file', 'missing_report', 'client_industry_unknown',
                   'tag_in_six_months',
                   'missing_years_on_file']:
        for func in categorical_columns_functions:
            print("%s for column %s:\r\n%s" % (func[0], column, func[1](client_data, column)))
    client_data = add_years_boolean(client_data)

    run_tests_on_data(client_data)




start = time.clock()
main()
elapsed = time.clock()
elapsed = elapsed - start
print ("Time spent in (Main) is: ", elapsed)