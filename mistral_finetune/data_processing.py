from utils import str_to_dict_eedi_df
import pandas as pd
import json

def process_data(data_address):
    data_file = pd.read_csv(data_address)
    data_file = str_to_dict_eedi_df(data_file)
    data = []
    for _, row in data_file.iterrows():
        temp = {}
        input  = '[INST] ' + "You are given the following math question along with the correct answer and explanation. Please use the following template to give three alternative incorrect answers to be used as multiple-choice options in a multiple-choice exam. Prior to the incorrect answer, provide feedback to be displayed to the student as an explanation of why that is not the correct answer.\n \
        [Template]\n \
        Distractor1 Feedback: \
        Distractor1: \
        Distractor2 Feedback: \
        Distractor2: \
        Distractor3 Feedback: \
        Distractor3:\n" +  "Question: " + row["question"].strip() + "\nExplanation: " + row["correct_option"]["explanation"].strip() + " \nAnswer: " + row["correct_option"]["option"].strip() + " [/INST]" 
        temp["input"] = input
        output = "\nDistractor1 Feedback: " + row["distractors"][0]["explanation"].strip() + "\nDistractor1: " + row["distractors"][0]["option"].strip() 
        output += "\nDistractor2 Feedback: " + row["distractors"][1]["explanation"].strip() + "\nDistractor2: " + row["distractors"][1]["option"].strip()
        output += "\nDistractor3 Feedback: " + row["distractors"][2]["explanation"].strip() + "\nDistractor3: " + row["distractors"][2]["option"].strip()
        temp["output"] = output
        data.append(temp)
    return data

test_data = process_data("data/eedi_test_20_cleaned_4_18.csv")
temp_data = process_data("data/eedi_train_80_cleaned_4_18.csv")
train_data = temp_data[:int(len(temp_data)*0.8)]
valid_data = temp_data[int(len(temp_data)*0.8):]

with open('data/train.json', 'w') as outfile:
    json.dump(train_data, outfile, indent=4)
        
with open('data/valid.json', 'w') as outfile:
    json.dump(valid_data, outfile, indent=4)

with open('data/test.json', 'w') as outfile:
    json.dump(test_data, outfile, indent=4)