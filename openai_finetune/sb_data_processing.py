import json
import random
import numpy as np
seed_val = 50
random.seed(seed_val)
import math
import re
import ast
import pandas as pd


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
    system_compoenet = {"role": "system", "content": "You are given the following math question. Please generate the correct answer."}
    user_component = {"role": "user", "content": "Question: " + row["question"].strip()} 
    assist_component = {"role": "assistant", "content": "Answer: " + row["correct_option"]["option"].strip()}
    data.append({"messages": [system_compoenet, user_component, assist_component]})
    
train_data = data[:200]
val_data = data[200:250]
    
data_file = pd.read_csv("data/eedi_test_20_cleaned_4_18.csv")
data_file = str_to_dict_eedi_df(data_file)
test_data = []
cor_answers = []
for i, row in data_file.iterrows():
    system_compoenet = {"role": "system", "content": "You are given the following math question. Please generate the correct answer."}
    user_component = {"role": "user", "content": "Question: " + row["question"].strip()} 
    cor_ans = row["correct_option"]["option"].strip()
    cor_answers.append(cor_ans)
    test_data.append([system_compoenet, user_component]) 
     

with open("data/train_cor_data.jsonl", "w") as outfile:
    for entry in train_data:
        json.dump(entry, outfile, ensure_ascii=False)
        outfile.write('\n')
        
with open("data/val_cor_data.jsonl", "w") as outfile:
    for entry in val_data:
        json.dump(entry, outfile, ensure_ascii=False)
        outfile.write('\n')
    
with open("data/test_cor_data.json", "w") as outfile:
        json.dump({"test_input": test_data}, outfile, ensure_ascii=False, indent=2)
        
with open("data/test_cor_answers.json", "w") as outfile:
        json.dump(cor_answers, outfile, ensure_ascii=False, indent=2)