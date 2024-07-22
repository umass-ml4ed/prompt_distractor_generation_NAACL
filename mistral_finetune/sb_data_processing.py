import json
import random
import pandas as pd
from utils import str_to_dict_eedi_df

seed_val = 50
random.seed(seed_val)

def process_data(data_address):
    data_file = pd.read_csv(data_address)
    data_file = str_to_dict_eedi_df(data_file)
    data = []
    for _, row in data_file.iterrows():
        temp = {}
        input = "[INST] You are given the following math question. Please generate the correct answer.\n" +\
            "Question: " + row["question"].strip() + "[/INST]"
        temp["input"] = input
        output = "\nAnswer: " + row["correct_option"]["option"].strip()
        temp["output"] = output
        data.append(temp)
    return data

test_data = process_data("data/eedi_test_20_cleaned_4_18.csv")
temp_data = process_data("data/eedi_train_80_cleaned_4_18.csv")
train_data = temp_data[:int(len(temp_data)*0.8)]
valid_data = temp_data[int(len(temp_data)*0.8):]
cor_answers = [
    row["correct_option"]["option"].strip()
    for _, row in str_to_dict_eedi_df(pd.read_csv("data/eedi_test_20_cleaned_4_18.csv")).iterrows()
]

with open('data/train_cor.json', 'w') as outfile:
    json.dump(train_data, outfile, indent=4)

with open('data/valid_cor.json', 'w') as outfile:
    json.dump(valid_data, outfile, indent=4)

with open('data/test_cor.json', 'w') as outfile:
    json.dump(test_data, outfile, indent=4)

with open("data/test_cor_answers.json", "w") as outfile:
    json.dump(cor_answers, outfile, ensure_ascii=False, indent=4)
