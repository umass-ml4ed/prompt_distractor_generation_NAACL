import pandas as pd
from utils import str_to_dict_eedi_df

train_data = pd.read_csv('data/eedi_train_80_cleaned_4_18.csv')
train_data = str_to_dict_eedi_df(train_data)
test_data = pd.read_csv('data/eedi_test_20_cleaned_4_18.csv')
test_data = str_to_dict_eedi_df(test_data)
all_data = pd.concat([train_data, test_data])

test_level_name = 'construct2' # construct1 construct2 construct3

# misconceptions = pd.read_csv('data/distractor_misconceptions.csv')
# questions = pd.read_csv('data/number_questions.csv')
# questions["QuestionId"] = questions["RealQuestionId"]

# output = pd.merge(misconceptions, questions, on='QuestionId', how='inner')

# sub_mis_mapping = {}
# level_name = "Level3SubjectName" # Level2SubjectName Level3SubjectName ConstructName
# for _, row in output.iterrows():
#     level_mis = sub_mis_mapping.setdefault(row[level_name], set())
#     level_mis.add(row['MisconceptionName'])

sub_mis_mapping = {}
for _, row in all_data.iterrows():
    level_mis = sub_mis_mapping.setdefault(row['construct_info'][test_level_name][1], set())
    for distractor in row["distractors"]:
        if distractor['misconception']:
            level_mis.add(distractor['misconception'])

number_mis_counter = 0
for i in sub_mis_mapping:
    print(i, len(sub_mis_mapping[i]))
    number_mis_counter += len(sub_mis_mapping[i])
print("Total misconceptions:", number_mis_counter)

counter = 0
for _, row in test_data.iterrows():
    if row['construct_info'][test_level_name][1] in sub_mis_mapping:
        counter += 1
        # create a new column called 'misconceptions' and add the list of misconceptions, else add empty list
        test_data.at[_, 'misconceptions'] = str(sub_mis_mapping[row['construct_info'][test_level_name][1]])
    else:
        test_data.at[_, 'misconceptions'] = str([])
print("Questions with misconceptions:", counter, "/", len(test_data))

# save the new data
test_data.to_csv('data/eedi_test_20_cleaned_4_18_misconceptions.csv', index=False)
