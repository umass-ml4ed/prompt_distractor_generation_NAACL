import pandas as pd
import ast
import re
from utils import initialize_seeds
import random
from utils import str_to_dict_eedi_df, clean_string, relaxed_metric, hard_metric, proportional_metric


def main():
    initialize_seeds(40)
    gt_distractors = []
    generated_distractors = []
    proportions = []
    distractor_pattern = re.compile(r"(?i)distractor ?(?:1|2|3): (.+)")
    
    data = pd.read_csv("") # get the file path in the analysis folder
    data = str_to_dict_eedi_df(data)
    for idx, row in data.iterrows():
        distractors_per_question = []
        response = row['raw_response']
        lines = response.split('\n')
        for line in lines:
            if distractor_pattern.match(line):
                distractor = distractor_pattern.match(line).group(1)
                distractors_per_question.append(clean_string(distractor))
        generated_distractors.append(distractors_per_question)
        
    gt_data = pd.read_csv("data/eedi_test_20_cleaned_4_18.csv")
    gt_data = str_to_dict_eedi_df(gt_data)
    for idx, row in gt_data.iterrows():
        gt_distractors.append([clean_string(row["distractors"][0]['option']), clean_string(row["distractors"][1]['option']), clean_string(row["distractors"][2]['option'])])
        proportions.append({clean_string(row["distractors"][0]['option']) : row["distractors"][0]['proportion'], clean_string(row["distractors"][1]['option']) : row["distractors"][1]['proportion'], clean_string(row["distractors"][2]['option']) : row["distractors"][2]['proportion']})
    
    print("Relaxed metric: ", relaxed_metric(gt_distractors, generated_distractors))
    print("Hard metric: ", hard_metric(gt_distractors, generated_distractors))
    print("Proportional metric: ", proportional_metric(gt_distractors, generated_distractors))
        
if __name__ == "__main__":
    main()


    