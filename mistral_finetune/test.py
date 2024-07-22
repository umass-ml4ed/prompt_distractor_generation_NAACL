from typing import List
import argparse
from tqdm import tqdm
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dataset import mistral_test_Dataset
from peft import PeftModel
from mistral_finetune.mistral_utils import test_tokenize, get_dataloader, BytesEncoder

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)

NUM_SB_SAMPLES = 20

def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, help="Path of saved model checkpoint")
    parser.add_argument("-LMN", "--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2") # mistralai/Mistral-7B-v0.1  mistralai/Mistral-7B-Instruct-v0.2
    parser.add_argument("-B", "--batch_size", type=int, default=8)
    parser.add_argument("-S", "--strategy", type=str, default="G")
    parser.add_argument("--sb", action="store_true", help="For sampling-based baseline")
    params = parser.parse_args()
    return params

def main():
    args = add_params()
    if args.sb:
        input_filename = "data/test_cor.json"
        output_filename = "mistral_SB_preproc.json"
    else:
        input_filename = "data/test.json"
        output_filename = "mistral_finetune.json"

    with open(input_filename, "r") as f:
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
        device_map="auto"
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    model = PeftModel.from_pretrained(base_model, args.model_checkpoint).to(device)
    model.eval()

    predictions: List[str] = []
    for step, batch in tqdm(enumerate(test_dataloader), total = len(test_dataloader)):
        input_ids, attn_masks = batch["input_ids"], batch["attn_mask"]
        input_ids, attn_masks = input_ids.to(device), attn_masks.to(device)
        with torch.no_grad():
            if args.sb:
                output = model.generate(
                    input_ids = input_ids,
                    attention_mask = attn_masks,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens = 20,
                    do_sample = True,
                    temperature = 1.0, 
                    top_p = 0.9,
                    num_return_sequences = NUM_SB_SAMPLES)
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
        predictions.extend(prediction)

    parsed_predictions = []
    for prediction in predictions:
        parsed_predictions.append(prediction.split("[/INST]")[1].strip())

    with open(output_filename, "w") as f:
        json.dump(parsed_predictions, f, cls=BytesEncoder, indent=4)

if __name__ == '__main__':
    main()
