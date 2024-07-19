import json
import random
import ast
import pandas as pd

seed_val = 50
random.seed(seed_val)

def str_to_dict_eedi_df(df):
	cols = ["correct_option", "distractors", "construct_info"]
	for i, row in df.iterrows():
		for col in cols:
			df.at[i, col] = ast.literal_eval(row[col])
	return df

data_file = pd.read_csv("data/eedi_train_80_cleaned_4_18.csv")
data_file = str_to_dict_eedi_df(data_file)
data = []
for i, row in data_file.iterrows():
    system_compoenet = {"role": "system", "content": "You are given the following math question along with the correct answer and explanation. Please use the following template to give three alternative incorrect answers to be used as multiple-choice options in a multiple-choice exam. Prior to the incorrect answer, provide feedback to be displayed to the student as an explanation of why that is not the correct answer.\n \
        [Template]\n \
        Distractor1 Feedback: \
        Distractor1: \
        Distractor2 Feedback: \
        Distractor2: \
        Distractor3 Feedback: \
        Distractor3:"}
    user_component = {"role": "user", "content": "Question: " + row["question"].strip() + "\nExplanation: " + row["correct_option"]["explanation"].strip() + " \nAnswer: " + row["correct_option"]["option"].strip()} 
    assist_component = {"role": "assistant", "content":  "Distractor1 Feedback: " + row["distractors"][0]["explanation"].strip() + "\nDistractor1: " + row["distractors"][0]["option"].strip() + "\nDistractor2 Feedback: " + row["distractors"][1]["explanation"].strip() + "\nDistractor2: " + row["distractors"][1]["option"].strip() + "\nDistractor3 Feedback: " + row["distractors"][2]["explanation"].strip() + "\nDistractor3: " + row["distractors"][2]["option"].strip()}
    data.append({"messages": [system_compoenet, user_component, assist_component]})

train_data = data[:200]
val_data = data[200:250]

# train_data = data[:int(len(data) * .8)]
# val_data = data[int(len(data) * .8):]

data_file = pd.read_csv("data/eedi_test_20_cleaned_4_18.csv")
data_file = str_to_dict_eedi_df(data_file)
test_data = []
for i, row in data_file.iterrows():
    system_compoenet = {"role": "system", "content": "You are given the following math question along with the correct answer and explanation. Please use the following template to give three alternative incorrect answers to be used as multiple-choice options in a multiple-choice exam. Prior to the incorrect answer, provide feedback to be displayed to the student as an explanation of why that is not the correct answer.\n \
        [Template]\n \
        Distractor1 Feedback: \
        Distractor1: \
        Distractor2 Feedback: \
        Distractor2: \
        Distractor3 Feedback: \
        Distractor3:"}
    user_component = {"role": "user", "content": "Question: " + row["question"].strip() + "\nExplanation: " + row["correct_option"]["explanation"].strip() + " \nAnswer: " + row["correct_option"]["option"].strip()}
    test_data.append([system_compoenet, user_component]) 


with open("data/train_data.jsonl", "w") as outfile:
    for entry in train_data:
        json.dump(entry, outfile, ensure_ascii=False)
        outfile.write('\n')

with open("data/val_data.jsonl", "w") as outfile:
    for entry in val_data:
        json.dump(entry, outfile, ensure_ascii=False)
        outfile.write('\n')

with open("data/test_data.json", "w") as outfile:
    json.dump({"test_input": test_data}, outfile, ensure_ascii=False, indent=2)
