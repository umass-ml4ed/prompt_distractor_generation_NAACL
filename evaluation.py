import pandas as pd
import re
from utils import initialize_seeds
import argparse
from utils import str_to_dict_eedi_df, clean_string, relaxed_metric, hard_metric, proportional_metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("num_distractors", type=int, default=3)
    args = parser.parse_args()

    initialize_seeds(40)
    gt_distractors = []
    generated_distractors = []
    proportions = []
    distractor_pattern = re.compile(r"(?i)distractor ?(?:\d+): (.+)")

    data = pd.read_csv(args.filename)
    data = str_to_dict_eedi_df(data)
    for idx, row in data.iterrows():
        distractors_per_question = []
        response = str(row['raw_response'])
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if distractor_pattern.match(line):
                distractor = distractor_pattern.match(line).group(1)
                distractors_per_question.append(clean_string(distractor))
            if distractors_per_question == args.num_distractors:
                break # Terminate early in case extra distractors generated
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
