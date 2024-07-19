import json
import numpy as np
import argparse
from utils import clean_string
from mistral_finetune.test import NUM_SB_SAMPLES

seed_val = 50
np.random.seed(seed_val)

parser = argparse.ArgumentParser()
parser.add_argument("num_distractors", type=int)
args = parser.parse_args()

with open('mistral_SB_preproc.json') as f:
    data = json.load(f)

with open("data/test_cor_answers.json") as f:
    cor_answers = json.load(f)

data = np.array(data).reshape((-1, NUM_SB_SAMPLES))
results = []
for idx, vals in enumerate(data):
    # Collect unique outputs that are not equal to the correct answer
    distractors = set()
    cor = clean_string(cor_answers[idx])
    for val in vals:
        if "Answer:" in val:
            val = val.split("Answer:")[1]
        if clean_string(val) == cor:
            continue
        distractors.add(val)
    # Keep necessary number of distractors and format to expected output
    distractors = list(distractors)[:args.num_distractors]
    result = ""
    for idx, val in enumerate(distractors):
        result += f"Distractor{idx+1}: {val}\n"
    results.append(result)

with open("mistral_SB.json", "w") as outfile:
    json.dump(results, outfile, ensure_ascii=False, indent = 2)
