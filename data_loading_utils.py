import json
import csv
import os.path
import pandas as pd
import time
import ast

NUM_OF_QUESTIONS = 1955

def cleanAndReformatData(cfg):
    print("Cleaning and reformatting input data...")
    data_tic = time.time()
    df = parse_eedi(
        csv_data=cfg.data.trainFilepath,
        exclude_same_construct=False,
        expand_distractors=cfg.command.task == "gen_feedback"
    )
    data_toc = time.time()
    print("Parsed", len(df), "examples in", data_toc - data_tic, "seconds.")
    return df

def parse_eedi(csv_data, exclude_same_construct=False, expand_distractors=False):
	# Cleans the unused entries from the dataset and reformats it.
	file_name = csv_data.split(".csv")[0]
	if exclude_same_construct:
		file_name += "_exclude_same_construct"
	if expand_distractors:
		file_name += "_expanded"
	# If json file doesn't exist for the csv, we create one here. 
	construct_list = []
	if not os.path.isfile(f"{file_name}.json"):
		qcnt = 0
		data = []
		with open(csv_data, encoding='utf-8') as f:
			reader = csv.DictReader(f)
			for rows in reader:
				qid = rows["id"]
				if rows["Cleaned"] == "TRUE" or rows["Script Clean"] == "TRUE":
					if rows["Discuss Flag"] == "FALSE":
						del rows["mathpix"], rows["Cleaned"], rows["Image Needed"], rows["Discuss Flag"], rows["Empty"], rows["Script Clean"], rows["Discussion Note"], rows["Pattern"]
						data_format = {
										"id": qid, \
										"question": rows["question"], \
										"correct_option": {
															"option_idx": int(rows['CorrectAnswer']), \
															"option": rows[f"Answer{rows['CorrectAnswer']}"], \
															"explanation": rows[f"Explanation{rows['CorrectAnswer']}"], \
															"proportion": float(rows[f"Answer{rows['CorrectAnswer']}Proportion"]),
														},
										"construct_info": {"construct1": [rows["Level2SubjectId"], rows["Level2SubjectName"]], \
														   "construct2": [rows["Level3SubjectId"], rows["Level3SubjectName"]], \
														   "construct3": [rows["ConstructId"], rows["ConstructName"]]
														}
										}
						if not exclude_same_construct or (exclude_same_construct and rows["ConstructId"] not in construct_list):
							construct_list.append(rows["ConstructId"])
							data_format["distractors"] = []
							dis_opts = ["1", "2", "3", "4"]
							dis_opts.remove(rows["CorrectAnswer"])
							for opt_idx in dis_opts:
								dis_info = 	{
												"option_idx": int(opt_idx), \
												"option": rows[f"Answer{opt_idx}"], \
												"explanation": rows[f"Explanation{opt_idx}"], \
												"proportion": float(rows[f"Answer{opt_idx}Proportion"]),
											}
								data_format["distractors"].append(dis_info)
							if expand_distractors:
								distractors = data_format["distractors"]
								del data_format["distractors"]
								for dis_info in distractors:
									data.append({**data_format, **dis_info})
									qcnt += 1
							else:
								data.append(data_format)
								qcnt += 1

					# else:
					# 	empty_dict = {"id": qid}
					# 	data.append(empty_dict)

			print(f"Total valid question count: {qcnt}")
			
			with open(f"{file_name}.json", 'w', encoding='utf-8') as f:
				f.write(json.dumps(data, indent=4))
			print("construct_list: ", len(construct_list))
			df = pd.read_json(f"{file_name}.json")
			return df
	else:
		df = pd.read_json(f"{file_name}.json")
		return df

# if __name__ == "__main__":
#     eedi_df = parse_eedi("./data/pp_eedi_data_0405.csv")
#     print(eedi_df.head())
