from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import argparse
import wandb
from tqdm import tqdm
import math
from utils import initialize_seeds
from mistral_finetune.mistral_utils import get_data, return_dl

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)

def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-W', '--wandb', action='store_true', help='For Wandb logging')
    parser.add_argument("-TMN", "--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2") # "google/flan-t5-large" "google/flan-t5-base"  "mistralai/Mistral-7B-v0.1"
    parser.add_argument("-TB", "--batch_size", type=int, default=16, help="Batch size for training T5")
    parser.add_argument("-LR", "--lr", type=float, default=3e-5, help="Learning Rate for T5")
    parser.add_argument("-GA", "--grad_acc", type=int, default=4, help="gradient accumulation steps")
    parser.add_argument("-E", "--num_epochs", type=int, default=8, help="Total Number of Epochs")
    parser.add_argument("-WD", "--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("-GC", "--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--sb", action="store_true", help="For sampling-based baseline")
    params = parser.parse_args()
    return params

if __name__ == '__main__':
    args = add_params()
    if args.wandb:
        wandb.init()
    initialize_seeds(221)

    save_name = f"./saved_models/finetune_mistral_{args.lr}_{args.num_epochs}_{args.weight_decay}_{args.grad_clip}_{args.sb}"
    print("Save model checkpoint to", save_name)
    least_val_loss = math.inf
    best_epoch = 0

    if args.sb:
        train_filename = "data/train_cor.json"
        val_filename = "data/valid_cor.json"
    else:
        train_filename = "data/train.json"
        val_filename = "data/valid.json"

    train_input, train_completions = get_data(train_filename)
    val_input, val_completions = get_data(val_filename)

    print(train_completions[0])
    print('________________'*5)
    print(val_completions[0]) 

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"   

    train_dl = return_dl(train_input, train_completions, tokenizer, args.batch_size, True)
    val_dl = return_dl(val_input, val_completions, tokenizer, args.batch_size, False)

    peft_config = LoraConfig(
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            r=32,
            lora_alpha=16,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            inference_mode=False,
        )
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                                      quantization_config=bnb_config,
                                                      pad_token_id=tokenizer.pad_token_id,
                                                      torch_dtype=torch.float32,
                                                      device_map="auto"
                                                      )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(base_model)
    model = get_peft_model(base_model, peft_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for i in range(args.num_epochs):
        model.train()
        total_train_loss = 0
        for step, batch in enumerate(tqdm(train_dl)):
            b_inpids, b_attnids, b_labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            model_outputs = model(input_ids=b_inpids, attention_mask=b_attnids, labels=b_labels)
            loss_batch = model_outputs.loss
            total_train_loss += loss_batch.item()
            loss_batch = loss_batch / args.grad_acc
            loss_batch.backward()
            if (step + 1) % args.grad_acc == 0 or step == len(train_dl) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
        avg_train_loss = total_train_loss / len(train_dl)
        print("Epoch: ", i, "Train Loss: ", avg_train_loss)

        model.eval()
        total_val_loss = 0
        for step, batch in enumerate(tqdm(val_dl)):
            b_inpids, b_attnids, b_labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            with torch.no_grad():
                model_outputs = model(input_ids=b_inpids, attention_mask=b_attnids, labels=b_labels)
                loss_batch = model_outputs.loss
            total_val_loss += loss_batch.item()
        avg_val_loss = total_val_loss / len(val_dl)
        print("Epoch: ", i, "Val Loss: ", avg_val_loss)
        if avg_val_loss < least_val_loss:
            print("Saving model")
            least_val_loss = avg_val_loss
            best_epoch = i
            model.save_pretrained(save_name)

        if args.wandb:
            wandb.log({
                "Average training loss": avg_train_loss,
                "Average validation loss":  avg_val_loss,
                "Best epoch": best_epoch})
