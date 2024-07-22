import csv
import re
import pandas as pd
import ast
import numpy as np
import torch
import random


def str_to_dict_eedi_df(df: pd.DataFrame):
    cols = ["correct_option", "gt_distractors", "generated_distractors", "distractors", "construct_info"]
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

    # Standardize symbols
    string = string.replace("\\%", "%")
    string = string.replace("...", "\\ldots")
    string = string.replace('÷', '\\div')
    string = string.replace('≥', '\\geq')
    string = string.replace('≤', '\\leq')
    string = string.replace('≠', '\\neq')
    string = string.replace('≈', '\\approx')
    string = string.replace('δ', '\\delta')
    string = string.replace('|', '\\vert')

    # Remove math environment indicators
    string = string.replace("$", "")
    string = string.replace("\\[", "")
    string = string.replace("\\]", "")
    string = string.replace("\\(", "")
    string = string.replace("\\)", "")

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
    file_prefix = f"{fine_tune_str}{cfg.prompt.type}_ndis{cfg.prompt.num_distractors}_{cfg.openAI.model}_{cfg.retriever.type}_{cfg.retriever.encodingPattern}{construct_str}"
    full_path = f"./analysis/{file_prefix}.csv"
    results.to_csv(full_path, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)

def relaxed_metric(gt_distractors, generated_distractors):
    # relaxed metric, if any gt_distractors are in the generated distractors, then it is correct
    correct = 0
    for gen_distractor_set, gt_distractor_set in zip(generated_distractors, gt_distractors):
        for generated_distractor in gen_distractor_set:
            if generated_distractor in gt_distractor_set:
                correct += 1
                break
    return correct/len(gt_distractors)

def hard_metric(gt_distractors, generated_distractors):
    # hard metric, if all gt_distractors are in the generated distractors, then it is correct
    correct = 0
    for gen_distractor_set, gt_distractor_set in zip(generated_distractors, gt_distractors):
        if all([gt_dis in gen_distractor_set for gt_dis in gt_distractor_set]):
            correct += 1
    return correct/len(gt_distractors)

def proportional_metric(gt_distractors, generated_distractors):
    correct = 0
    for gen_distractor_set, gt_distractor_set in zip(generated_distractors, gt_distractors):
        props = 0
        for gt_distractor in gt_distractor_set:
            if gt_distractor in gen_distractor_set:
                props += 1
        correct += props/3
    return correct/len(gt_distractors)
