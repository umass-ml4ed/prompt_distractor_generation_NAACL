# Imports 
import argparse
from tqdm import tqdm
import numpy as np
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils import test_tokenize, get_dataloader, BytesEncoder
from dataset import mistral_test_Dataset
from peft import PeftModel
import csv
import pandas as pd

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)

def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("-LMN", "--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2") # mistralai/Mistral-7B-v0.1  mistralai/Mistral-7B-Instruct-v0.2
    parser.add_argument("-B", "--batch_size", type=int, default=8)
    parser.add_argument("-S", "--strategy", type=str, default="B")
    params = parser.parse_args()
    return params

def main():
    args = add_params()
    data_name = "test"
    with open(f"data/{data_name}.json", "r") as f:
        data = json.load(f)
        inputs = []
        for i in data:
            inputs.append(i["input"])
        
    # llama tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    # tokenize
    input_ids, attn_masks = test_tokenize(tokenizer, inputs)
    # create dataset
    test_dataset = mistral_test_Dataset(input_ids, attn_masks)
    test_dataloader = get_dataloader(args.batch_size, test_dataset, False)
    
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        pad_token_id=tokenizer.pad_token_id,
        quantization_config=bnb_config,
        torch_dtype=torch.float32,
        device_map={"": 0}
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1
    
    model = PeftModel.from_pretrained(base_model, "./saved_models/finetune_mistral_8e-05_8_0.0_1.0").to(device)
    model.eval()
    
    predictions = []
    for step, batch in tqdm(enumerate(test_dataloader), total = len(test_dataloader)):
        input_ids, attn_masks = batch["input_ids"], batch["attn_mask"]
        input_ids, attn_masks = input_ids.to(device), attn_masks.to(device)
        with torch.no_grad():
            if args.strategy == "NS":
                output = model.generate(
                    input_ids = input_ids,
                    attention_mask = attn_masks,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens = 350,
                    do_sample = True,
                    temperature = 1.0, 
                    top_p = 0.9,
                    num_return_sequences = 5)
            elif args.strategy == "B":
                output = model.generate(
                    input_ids = input_ids,
                    attention_mask = attn_masks,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens = 350,
                    do_sample = False,
                    num_beams = 5,
                    num_return_sequences = 1)
            elif args.strategy == "G":
                output = model.generate(
                    input_ids = input_ids,
                    attention_mask = attn_masks,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens = 350,
                    do_sample = False,
                    num_beams = 1,
                    num_return_sequences = 1)
                
        prediction = tokenizer.batch_decode(output, skip_special_tokens=True)
        predictions = predictions + prediction
    
    parsed_predictions = []
    for prediction in predictions:
        parsed_predictions.append(prediction.split("[/INST]")[1].strip())
        
    # save predictions
    result_address = f"./predictions/mistral_finetune.json"
    with open(result_address, "w") as f:
        json.dump(parsed_predictions, f, cls=BytesEncoder, indent=4)

if __name__ == '__main__':
    main()