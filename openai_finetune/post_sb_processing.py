import json
import numpy as np

seed_val = 50
np.random.seed(seed_val)

with open('SB_samplingpre_proc.json') as f:
    data = json.load(f)

with open("data/test_cor_answers.json") as f:
    cor_answers = json.load(f)

parsed_data = []
for val in data:
    val = val.strip().replace("$", "").replace(",", "")
    if "The correct answer is " in val:
        val = val.split("The correct answer is ")[1]
    elif "The correct answer: " in val:
        val = val.split("The correct answer: ")[1]
    elif "Correct answer: " in val:
        val = val.split("Correct answer: ")[1]
    elif "Answer: " in val:
        val = val.split("Answer: ")[1]
    elif "Answer:" in val:
        val = val.split("Answer:")[1]
    elif "Answer=" in val:
        val = val.split("Answer=")[1]
    elif "answer: " in val:
        val = val.split("answer: ")[1]
    elif "Workout: " in val:
        val = val.split("Workout: ")[1]
    elif "workout: " in val:
        val = val.split("workout: ")[1]
    elif "Answer : " in val:
        val = val.split("Answer : ")[1]
    else:
        val = val
    parsed_data.append(val)

parsed_data = np.array(parsed_data)
parsed_data = np.reshape(parsed_data, (-1, 20))
final_datas = []
for idx, val in enumerate(parsed_data):
    # delete all the ones that are same with the correct answer
    val = list(set(val))
    if cor_answers[idx] in val:
        val.remove(cor_answers[idx])
    if len(list(set(val))) > 3:
        final_datas.append(list(set(val))[0:3])
    else:
        final_datas.append(list(set(val)))

results = []
for final_data in final_datas:
    result = ""
    for idx, val in enumerate(final_data):
        result += f"Distractor{idx+1}: {val}\n"
    results.append(result)

with open("SB_sampling.json", "w") as outfile:
    json.dump(results, outfile, ensure_ascii=False, indent = 2)
