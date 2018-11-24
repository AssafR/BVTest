import pandas as pd
import re as re
import itertools
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt

pd.options.display.max_columns = 999
p = re.compile('between (.*?) and (.*?) years')

frame_statistic_functions = [
    ["Head", lambda frame: frame.head()],
    ["Data Description", lambda frame: frame.describe()],
    ["Number of NaN values", lambda frame: frame.isna().sum()],
    ["Number of unique values", lambda frame: frame.nunique()]
]

column_statistic_functions = [
    ["value counts", lambda frame, column: frame[column].value_counts()]
]

categorical_columns_functions = [
    ["unique values", lambda frame, column: frame[column].unique()]
]

client_data = pd.read_csv("./data_science_exam_bluevine_file.csv")

print('Some Data exploration')
for func in frame_statistic_functions:
    print("\r\n%s:\r\n%s" % (func[0], func[1](client_data)))

for func in column_statistic_functions:
    print("\r\n%s:\r\n%s" % (func[0], func[1](client_data, 'period')))

for column in ['period', 'segment', 'years_on_file', 'missing_report', 'client_industry_unknown', 'tag_in_six_months']:
    for func in categorical_columns_functions:
        print("%s for column %s:\r\n%s\r\n" % (func[0], column, func[1](client_data, column)))

# New column indicating there is no data in this important field
client_data["no_years_on_file"] = client_data["years_on_file"].isnull().map({True:1, False:0})
# Change the missing values into a standard form
client_data.fillna(value={'years_on_file': "between 0 and 0 years"},inplace=True)

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
for column in ['period', 'segment', 'years_on_file', 'missing_report', 'client_industry_unknown', 'tag_in_six_months',
               'no_years_on_file']:
    for func in categorical_columns_functions:
        print("%s for column %s:\r\n%s\r\n" % (func[0], column, func[1](client_data, column)))

years_on_file_bands = ["0","0.5","1","1.5","2","2.5","3","5","6","8","10","12","14","17","20","25","30","35","50"]
columns_for_years_bands = ["years_file_over_" + x for x in years_on_file_bands]

#years_column =

def get_two_indices_from_string(source_row):
    str = source_row['years_on_file']
    match = p.search(str)
    max_index = years_on_file_bands.index(match.group(2)) +1
    new_array = [0]* len(years_on_file_bands)
    new_array[0:max_index] = [1]*max_index
    row = dict(zip(years_on_file_bands,new_array))
    return row

df2 = pd.DataFrame(index=client_data.index,columns = years_on_file_bands )

print("Calculating")
for (idx, row) in client_data.iterrows():
    print(idx)
    x = get_two_indices_from_string(row)
    df2.iloc[idx] = x

print("Joining",df2.shape,client_data.shape)
client_data = client_data.join(df2)
print(client_data.head())

#client_data['pop']= client_data['continent'].map(pop_dict)


#print(client_data.head(10))
# Enumerate categorical text fields, e.g. field 'tag_in_six_months': enumerate ['bad' 'good']
# period, segment, tag_in_six_months
