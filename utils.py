import csv
import re
import pandas as pd
import ast
import numpy as np
import torch
import random


def str_to_dict_eedi_df(df: pd.DataFrame):
    cols = ["correct_option", "gt_distractors", "generated_distractors", "distractors"]
    cols = [col for col in cols if col in df.columns]
    for i, row in df.iterrows():
        for col in cols:
            try:
                df.at[i, col] = ast.literal_eval(row[col])
            except Exception:
                df.at[i, col] = None
    return df

def initialize_seeds(seed_num: int):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        
def clean_string(string):
    string = string.lower()
    string = string.replace("$", "")
    string = string.replace("£", "")
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = string.replace("...", "\\ldots")
    string = string.replace('÷', '\\div')
    string = string.replace('≥', '\\geq')
    string = string.replace('≤', '\\leq')
    string = string.replace('≠', '\\neq')
    string = string.replace('≈', '\\approx')
    string = string.replace('δ', '\\delta')
    string = string.replace('|', '\\vert')
    string = string.replace(" hours", "")
    string = string.replace("\\(\\frac", "\\frac")
    string = string.replace("}\\)", "}")
    string = string.replace("\\[", "")
    string = string.replace("\\]", "")
    if string == "both towers are the same height":
        string = "$a \\& b$ are the same height"
    if string == 'square root of 96':
        string = '\\sqrt{96}'
    if string == "\\frac{2.0}{3.0}":
        string = "\\frac{2}{3}"

    # convert / and \div fractions to \frac
    string = re.sub(r"([\d\.]+)\s*(/|\\div)\s*([\d\.]+)", r"\\frac{\g<1>}{\g<3>}", string) 
    # convert x to \times
    string = re.sub(r'\s*×\s*', r' \\times ', string)
    # convert √ to \\sqrt{}
    string = re.sub(r'√', r'\\sqrt', string) 
    # convert 2 cm to 2 \mathrm{~cm}
    string = re.sub(r'(\d+(?:\.\d+)?)\s*cm',  r'\1 \\mathrm{~cm}', string)
    # convert 2 m to 2 \mathrm{~m}
    string = re.sub(r'(\d+(?:\.\d+)?)\s*m',  r'\1 \\mathrm{~m}', string)
    # convert 2 km to 2 mathrm{~km}
    string = re.sub(r'(\d+(?:\.\d+)?)\s*km',  r'\1 \\mathrm{~km}', string)

    # convert p^2 to p^{2}
    string = re.sub(r'([a-zA-Z])\^(\d+)', r'\1^{\2}', string)
    
    # remove hyphen between words
    string = re.sub(r'([a-zA-Z]+)-([a-zA-Z]+)', r'\1\2', string)
    
    string = string.replace('\\mathrm{~m}athrm{~cm}', '\\mathrm{~cm}')
    string = string.replace('\\mathrm{~m}ore', 'more')
    string = string.replace(' ', '')
    string = string.strip()
    return string
    
def save_result(results, cfg):
    """
    Saves the results to a csv file.
    returns: the filepath of the saved file.
    """
    construct_str = "" if not cfg.retriever.exclude_construct_level else f"_{cfg.retriever.exclude_construct_level}"
    fine_tune_str = "" if not cfg.dir_finetune_result.model_name else f"{cfg.dir_finetune_result.model_name}_"
    file_prefix = f"{fine_tune_str}{cfg.prompt.type}_{cfg.openAI.model}_{cfg.retriever.type}_{cfg.retriever.encodingPattern}{construct_str}"
    full_path = f"./analysis/{file_prefix}.csv"
    results.to_csv(full_path, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
    
def relaxed_metric(gt_distractors, generated_distractors):
    # relaxed metric, if any gt_distractors are in the generated distractors, then it is correct
    correct = 0
    for idx, gt_distractor in enumerate(gt_distractors):
        for generated_distractor in generated_distractors[idx]:
            if generated_distractor in gt_distractor:
                correct += 1
                break
    return correct/len(gt_distractors)

def hard_metric(gt_distractors, generated_distractors):
    # hard metric, if all gt_distractors are in the generated distractors, then it is correct
    correct = 0
    for idx, gt_distractor in enumerate(gt_distractors):
        if gt_distractor[0] in generated_distractors[idx] and gt_distractor[1] in generated_distractors[idx] and gt_distractor[2] in generated_distractors[idx]:
            correct += 1
    return correct/len(gt_distractors)

def proportional_metric(gt_distractors, generated_distractors):
    correct = 0
    for idx, generated_distractor in enumerate(generated_distractors):
        props = 0
        for gt_distractor in gt_distractors[idx]:
            if gt_distractor in generated_distractor:
                props += 1
        correct += props/3
    return correct/len(gt_distractors)