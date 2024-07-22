from torch.utils.data import DataLoader
import torch
import json
from dataset import mistral_Dataset

def get_data(data_address):
    with open(data_address) as f:
        data = json.load(f)
        prompts = []
        completions = []
        for i in data:
            prompts.append(i["input"])
            completions.append(i["input"] + i["output"])
    return prompts, completions

# llama tokenizer
def tokenize(tokenizer, prompts, completions):
    prompts_tokenized = tokenizer(prompts, padding=False, truncation=True, max_length=2048)
    # add eos token to every completion
    completions = [completion + tokenizer.eos_token for completion in completions]
    completions_tokenized = tokenizer(completions, padding=True, truncation=True, max_length=2048, return_tensors='pt')
    # Construct labels
    labels = completions_tokenized["input_ids"].detach().clone()
    # Ignore pad tokens when computing loss
    labels = labels.masked_fill((completions_tokenized["attention_mask"] == 0), -100)
    # Ignore prompt tokens when computing loss
    prompts_len = torch.tensor([len(prompt_tokenized_input_ids) for prompt_tokenized_input_ids in prompts_tokenized["input_ids"]])
    range_tensor = torch.arange(completions_tokenized["input_ids"].size(1)).unsqueeze(0)
    range_tensor = range_tensor.repeat(prompts_len.size(0), 1)
    mask_tensor = (range_tensor < prompts_len.unsqueeze(-1))
    labels[mask_tensor] = -100
    return completions_tokenized["input_ids"], completions_tokenized["attention_mask"], labels

def return_dl(prompts, completions, tokenizer, batch_size, shuffle):
    llama_input_ids, llama_attn_masks, llama_labels = tokenize(tokenizer, prompts, completions)
    dataset = mistral_Dataset(llama_input_ids, llama_attn_masks, llama_labels)
    dataloader = get_dataloader(batch_size, dataset, shuffle)
    return dataloader

def test_tokenize(tokenizer, prompts):
    prompts_tokenized = tokenizer(prompts, padding=True, truncation=True, max_length=2048, return_tensors='pt')
    return prompts_tokenized["input_ids"], prompts_tokenized["attention_mask"]

def get_dataloader(batch_size, dataset, shuffle = False):
    return DataLoader(dataset = dataset, batch_size = batch_size, shuffle = shuffle)

class BytesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        return json.JSONEncoder.default(self, obj)
