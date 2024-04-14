import argparse
import random
import re
from typing import List
from itertools import combinations
import json
import ast
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_seeds(seed_num: int):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)

pred_re = re.compile(r"Preferred Answer: (A|B|Tie)$")

peft_config = LoraConfig(
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    inference_mode=False,
)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

def str_to_dict_eedi_df(df: pd.DataFrame):
    cols = ["correct_option", "distractors", "construct_info", "pred_dis", "pred_f"]
    cols = [col for col in cols if col in df.columns]
    for i, row in df.iterrows():
        for col in cols:
            try:
                df.at[i, col] = ast.literal_eval(row[col])
            except Exception:
                df.at[i, col] = None
    return df

def normalize_candidate_distractors(gold_df: pd.DataFrame, df: pd.DataFrame):
    df = str_to_dict_eedi_df(df)
    df["distractors"] = None
    cases_with_fail = 0
    for i, row in df.iterrows():
        if not row["pred_dis"]:
            distractors = [None] * 3
            explanations = [None] * 3
        else:
            distractors = []
            explanations = []
            for di in range(3):
                if row["pred_dis"][di] not in distractors and row["pred_dis"][di] != gold_df.iloc[i]["correct_option"]["option"]:
                    distractors.append(row["pred_dis"][di])
                    explanations.append(row["pred_f"][di])
            distractors += [None] * (3 - len(distractors))
            explanations += [None] * (3 - len(explanations))
        if None in distractors:
            cases_with_fail += 1
        df.at[i, "distractors"] = [{"option": d, "explanation": e} for d, e in zip(distractors, explanations)]
    print(f"Failed cases: {cases_with_fail}/{len(df)} ({100 * cases_with_fail / len(df):.2f}%)")
    return df

def compute_ranking_pairs(df: pd.DataFrame, args):
    return [
        {
            "question": row["question"],
            "correct_answer": row["correct_option"]["option"],
            "solution": row["correct_option"]["explanation"],
            "options": [row["distractors"][idx0]["option"], row["distractors"][idx1]["option"]],
            "explanations": [row["distractors"][idx0]["explanation"], row["distractors"][idx1]["explanation"]],
            "proportions": [row["distractors"][idx0]["proportion"], row["distractors"][idx1]["proportion"]],
            "a_wins": row["distractors"][idx0]["proportion"] > row["distractors"][idx1]["proportion"],
        }
        for _, row in df.iterrows()
        for idxs in combinations(range(3), 2)
        for idx0, idx1 in ([idxs, reversed(idxs)] if args.both_orders else [random.sample(idxs, 2)])
    ]

def compute_cross_df_ranking_pairs(gold_df: pd.DataFrame, cand_df: pd.DataFrame):
    return [
        {
            "question": gold_row["question"],
            "correct_answer": gold_row["correct_option"]["option"],
            "solution": gold_row["correct_option"]["explanation"],
            "options": [a_row["distractors"][a_idx]["option"], b_row["distractors"][b_idx]["option"]],
            "explanations": [a_row["distractors"][a_idx]["explanation"], b_row["distractors"][b_idx]["explanation"]],
        }
        for (_, gold_row), (_, cand_row) in zip(gold_df.iterrows(), cand_df.iterrows())
        for a_idx in range(3)
        for b_idx in range(3)
        for a_row, b_row in [(gold_row, cand_row), (cand_row, gold_row)]
    ]

def get_ranking_pairs(args):
    if args.test:
        df = str_to_dict_eedi_df(pd.read_csv("data/eedi_test_20_cleaned_4_18.csv"))
        return compute_ranking_pairs(df, args)

    if args.compare_methods:
        test_df = str_to_dict_eedi_df(pd.read_csv("data/eedi_test_20_cleaned_4_18.csv"))
        comp_df = normalize_candidate_distractors(test_df, pd.read_csv(args.compare_methods))
        return compute_cross_df_ranking_pairs(test_df, comp_df)

    df = str_to_dict_eedi_df(pd.read_csv("data/eedi_train_80_cleaned_4_18.csv"))
    if args.no_val:
        return compute_ranking_pairs(df, args), None

    df = df.sample(frac=1, random_state=221)
    return (
        compute_ranking_pairs(df.iloc[:int(len(df) * 0.8)], args),
        compute_ranking_pairs(df.iloc[int(len(df) * 0.8):], args)
    )

def get_task_prompt(ranking_pair: dict, args):
    return "A teacher assigns the following math multiple choice question to a class of middle school students.\n\n" +\
        f"Question: {ranking_pair['question']}\n" +\
        f"Correct Answer: {ranking_pair['correct_answer']}\n" +\
        f"Solution: {ranking_pair['solution']}\n\n" +\
        "Here are two incorrect options that some students choose:\n" +\
        f"Option A: {ranking_pair['options'][0]}\n" + (f"Option A Explanation: {ranking_pair['explanations'][0]}\n" if args.input_exp else "") +\
        f"Option B: {ranking_pair['options'][1]}\n" + (f"Option B Explanation: {ranking_pair['explanations'][1]}\n" if args.input_exp else "") +\
        "Which incorrect option are the students more likely to pick?\n"

def get_task_completion(reasoning: str, winner: str):
    return ((reasoning + "\n") if reasoning else "") + f"Preferred Answer: {winner}"

def get_winner(ranking_pair: dict, args):
    if abs(ranking_pair["proportions"][0] - ranking_pair["proportions"][1]) < args.tie_cutoff:
        return "Tie"
    return "A" if ranking_pair["proportions"][0] > ranking_pair["proportions"][1] else "B"

class RankingDataset(Dataset):
    def __init__(self, ranking_pairs: List[dict], test: bool, args):
        super().__init__()
        self.data = []
        label_to_count = {"A": 0, "B": 0, "Tie": 0}

        for idx, rp in enumerate(ranking_pairs):
            prompt = get_task_prompt(rp, args)
            if not args.compare_methods:
                winner = get_winner(rp, args)
                label_to_count[winner] += 1

            if args.finetune:
                if args.out_type == "exp":
                    reasoning = f"Option A Explanation: {rp['explanations'][0]}\nOption B Explanation: {rp['explanations'][1]}"
                else:
                    reasoning = ""
                self.data.append({
                    **rp,
                    "prompt": prompt,
                    "completion": get_task_completion(reasoning, winner)
                })

            elif test:
                self.data.append({**rp, "prompt": prompt})

        if not args.compare_methods:
            print(", ".join([f"{k}: {100 * v / len(self.data):.2f}%" for k, v in label_to_count.items()]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class RankingCollator:
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, batch: List[dict]):
        all_prompts = [sample["prompt"] for sample in batch]
        prompts_tokenized = self.tokenizer(all_prompts, return_tensors="pt", padding=True).to(device)
        if batch[0].get("completion"):
            all_inputs = [sample["prompt"] + sample["completion"] + self.tokenizer.eos_token for sample in batch]
            inputs_tokenized = self.tokenizer(all_inputs, return_tensors="pt", padding=True).to(device)
            prompt_lens = prompts_tokenized.attention_mask.sum(dim=1)
            labels = inputs_tokenized.input_ids.clone()
            padding_mask = torch.arange(labels.shape[1]).repeat(labels.shape[0], 1).to(device) < prompt_lens.unsqueeze(1)
            labels[padding_mask] = -100
            labels = labels.masked_fill(inputs_tokenized.attention_mask == 0, -100)
        else:
            inputs_tokenized = prompts_tokenized
            labels = None

        return {
            "input_ids": inputs_tokenized.input_ids,
            "attention_mask": inputs_tokenized.attention_mask,
            "labels": labels,
            "meta_data": batch
        }

def get_dataloader(dataset: RankingDataset, tokenizer, shuffle: bool, args):
    return DataLoader(dataset, collate_fn=RankingCollator(tokenizer, args), batch_size=args.batch_size, shuffle=shuffle)

def get_base_model(tokenizer, test: bool, args):
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        pad_token_id=tokenizer.pad_token_id,
        quantization_config=None if test else bnb_config,
        # Higher precision for non-quantized parameters helps training accuracy and doesn't hurt performance
        # Lower precision at test time improves speed and only marginally hurts performance
        torch_dtype=torch.float16 if test else torch.float32,
        device_map={"": 0}
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1
    return base_model

def get_model(test: bool, args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base_model = get_base_model(tokenizer, test, args)
    if test:
        model = PeftModel.from_pretrained(base_model, args.model_name).merge_and_unload()
    else:
        model = prepare_model_for_kbit_training(base_model)
        model = get_peft_model(model, peft_config)
    return model, tokenizer

def get_loss(batch, model):
    model_outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
    return model_outputs.loss

def train(args):
    # Load model
    model, tokenizer = get_model(False, args)

    # Load data
    train_data, val_data = get_ranking_pairs(args)
    if not args.no_val:
        val_dataset = RankingDataset(val_data, False, args)
        val_dataloader = get_dataloader(val_dataset, tokenizer, False, args)
    train_dataset = RankingDataset(train_data, False, args)
    train_dataloader = get_dataloader(train_dataset, tokenizer, True, args)

    # Train loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val_loss = None
    for _ in range(args.epochs):
        total_train_loss = 0
        total_val_loss = 0
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            loss = get_loss(batch, model)
            total_train_loss += loss.item()
            loss = loss / args.grad_accum_steps
            loss.backward()
            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        if not args.no_val:
            with torch.no_grad():
                model.eval()
                for batch in tqdm(val_dataloader):
                    loss = get_loss(batch, model)
                    total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader) if not args.no_val else 0
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if args.no_val or best_val_loss is None or avg_val_loss < best_val_loss:
            print("Best! Saving model...")
            model.save_pretrained(args.model_name)
            best_val_loss = avg_val_loss

def evaluate(model: AutoModelForCausalLM, tokenizer, dataloader: DataLoader, split: str, args):
    model.eval()
    tokenizer.padding_side = "left"
    label_to_val = {"A": 1, "B": 0, "Tie": 0.5, None: 0.5}
    with torch.no_grad():
        total_correct = 0
        total_mse = 0
        num_samples = 0
        all_results = []
        for batch in tqdm(dataloader):
            ouptut_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_gen_tokens,
                do_sample=False,
                num_beams=args.num_beams
            )
            preds = tokenizer.batch_decode(ouptut_ids, skip_special_tokens=True)
            final_preds = []
            for pred in preds:
                pred_match = pred_re.search(pred)
                final_preds.append(pred_match and pred_match.group(1))
            labels = [get_winner(sample, args) for sample in batch["meta_data"]]
            batch_correct = np.array([pred == label for pred, label in zip(final_preds, labels)])
            total_correct += batch_correct.sum()
            total_mse += sum([(label_to_val[pred] - label_to_val[label]) ** 2 for pred, label in zip(final_preds, labels)])            
            num_samples += len(batch["meta_data"])
            all_results += [
                {
                    "prompt": sample["prompt"],
                    "pred": pred,
                    "final_pred": final_pred,
                    "label": label,
                    "proportions": sample["proportions"]
                }
                for sample, pred, final_pred, label in zip(batch["meta_data"], preds, final_preds, labels)
            ]
    accuracy = total_correct / num_samples
    rmse = (total_mse / num_samples) ** 0.5
    with open(f"distractors/ranking_results_{split}_{args.model_name}_b{args.num_beams}.json", "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "rmse": rmse,
            "results": all_results
        }, f, indent=2, ensure_ascii=False)
    tokenizer.padding_side = "right"
    return accuracy, rmse

def test(args):
    model, tokenizer = get_model(True, args)
    data = get_ranking_pairs(args)
    test_dataset = RankingDataset(data, True, args)
    test_dataloader = get_dataloader(test_dataset, tokenizer, False, args)
    test_accuracy, test_rmse = evaluate(model, tokenizer, test_dataloader, "test", args)
    print(f"Accuracy: {test_accuracy:.4f}, RMSE: {test_rmse:.4f}")

def analyze_test(args):
    with open(args.analyze_test) as f:
        results = json.load(f)["results"]
    cutoff_to_total = {cutoff: 0 for cutoff in [0, 3, 5, 10, 15, 20]}
    cutoff_to_correct = cutoff_to_total.copy()
    for result in results:
        for cutoff in cutoff_to_total:
            if abs(result["proportions"][0] - result["proportions"][1]) > cutoff:
                cutoff_to_total[cutoff] += 1
                if result["label"] == result["final_pred"]:
                    cutoff_to_correct[cutoff] += 1
    for cutoff, total in cutoff_to_total.items():
        print(f"{cutoff} - Correct: {100 * cutoff_to_correct[cutoff] / total:.2f}%, Portion: {100 * total / len(results):.2f}%")

def compare_methods(args):
    assert args.batch_size % 2 == 0, "Need an even sized batch to identify gold vs candidate distractors"
    model, tokenizer = get_model(True, args)
    data = get_ranking_pairs(args)
    dataset = RankingDataset(data, True, args)
    dataloader = get_dataloader(dataset, tokenizer, False, args)
    model.eval()
    tokenizer.padding_side = "left"
    with torch.no_grad():
        total_score = 0
        all_results = []
        for batch in tqdm(dataloader):
            ouptut_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_gen_tokens,
                do_sample=False,
                num_beams=args.num_beams
            )
            preds = tokenizer.batch_decode(ouptut_ids, skip_special_tokens=True)
            final_preds = []
            for pred in preds:
                pred_match = pred_re.search(pred)
                final_preds.append(pred_match and pred_match.group(1))
            # Even entries give gold df A and odd give candidate df A - switch value of every other result
            a_wins = [pred == "A" for pred in final_preds]
            gold_wins = np.logical_xor(a_wins, np.arange(len(a_wins)) % 2)
            missing = np.array([not all(sample["options"]) for sample in batch["meta_data"]])
            ties = np.array([sample["options"][0] == sample["options"][1] for sample in batch["meta_data"]])
            scores = .5 * ties + (~gold_wins & ~ties & ~missing)
            total_score += scores.sum()
            all_results += [
                {
                    "prompt": sample["prompt"],
                    "pred": pred,
                    "final_pred": final_pred,
                    "score": score,
                }
                for sample, pred, final_pred, score in zip(batch["meta_data"], preds, final_preds, scores)
            ]
    avg_score = total_score / len(dataset)
    with open(f"distractors/ranking_results_comp_{args.model_name}_b{args.num_beams}.json", "w", encoding="utf-8") as f:
        json.dump({
            "score": avg_score,
            "results": all_results
        }, f, indent=2, ensure_ascii=False)
    print(f"Score: {avg_score:.4f}")

def main():
    initialize_seeds(221)

    parser = argparse.ArgumentParser()
    # Modes
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--analyze_test", type=str, help="Input name of test result file to analyze")
    parser.add_argument("--compare_methods", type=str, help="Input name of generation result file to compare")
    # Settings
    parser.add_argument("--both_orders", action="store_true")
    parser.add_argument("--input_exp", action="store_true")
    parser.add_argument("--out_type", choices=["exp", "final"], default="final")
    parser.add_argument("--tie_cutoff", type=float, default=0)
    parser.add_argument("--num_completions", type=int, default=1)
    parser.add_argument("--no_val", action="store_true")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_gen_tokens", type=int, default=300)
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--pt_model_name", type=str, default="distrank-ft-cot")
    parser.add_argument("--model_name", type=str, default="distrank-ft-cot")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    if args.finetune:
        train(args)
    if args.test:
        test(args)
    if args.analyze_test:
        analyze_test(args)
    if args.compare_methods:
        compare_methods(args)

if __name__ == "__main__":
    main()
