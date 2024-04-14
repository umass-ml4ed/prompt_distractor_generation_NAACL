import pandas as pd
import json
import ast
from utils import str_to_dict_eedi_df

misconceptions = pd.read_csv('data/distractor_misconceptions.csv') 
questions = pd.read_csv('data/number_questions.csv') 

output = pd.merge(misconceptions, questions,  
                   on='QuestionId',  
                   how='inner')

sub_mis_mapping = {}
level_name = "Level3SubjectName" # Level2SubjectName Level3SubjectName ConstructName
for _, row in output.iterrows():
    if row[level_name] not in sub_mis_mapping:
        sub_mis_mapping[row[level_name]] = []
    sub_mis_mapping[row[level_name]].append(row['MisconceptionName'])

# make sure there are no duplicates
for i in sub_mis_mapping:
    sub_mis_mapping[i] = list(set(sub_mis_mapping[i]))

number_mis_counter = 0
for i in sub_mis_mapping:
    print(i, len(sub_mis_mapping[i]))
    number_mis_counter += len(sub_mis_mapping[i])
print(number_mis_counter)
    
test_data = pd.read_csv('data/eedi_test_20_cleaned_4_18.csv')
test_data = str_to_dict_eedi_df(test_data)
counter = 0
test_level_name = 'construct2' # construct1 construct2 construct3
for _, row in test_data.iterrows():
    if row['construct_info'][test_level_name][1] in sub_mis_mapping:
        counter += 1
        # create a new column called 'misconceptions' and add the list of misconceptions, else add empty list
        test_data.at[_, 'misconceptions'] = str(sub_mis_mapping[row['construct_info'][test_level_name][1]])
    else:
        test_data.at[_, 'misconceptions'] = str([])

print(counter)

# save the new data
test_data.to_csv('data/eedi_test_20_cleaned_4_18_misconceptions.csv', index=False)



