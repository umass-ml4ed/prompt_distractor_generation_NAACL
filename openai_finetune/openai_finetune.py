import os
from openai import OpenAI
import json
from tqdm import tqdm

api_keys = os.getenv("OPENAI_API_KEYS").split(",") if "OPENAI_API_KEYS" in os.environ else [os.getenv("OPENAI_API_KEY")] 
client = OpenAI(api_key = api_keys)
train_file = client.files.create(
  file=open("data/train_data.jsonl", "rb"),
  purpose="fine-tune"
)
train_file_id = train_file.id
val_file = client.files.create(
    file=open("data/val_data.jsonl", "rb"),
    purpose="fine-tune"
    )
val_file_id = val_file.id

client.fine_tuning.jobs.create(
  training_file=train_file_id, 
  validation_file=val_file_id,
  model="gpt-3.5-turbo-1106",
  hyperparameters={"batch_size": 8, "n_epochs": 2})

# run on the test data, this code can be used for both FT and SB
responses = []
with open("data/test_data.json", "r") as f:
    test_data = json.load(f)
    test_questions = test_data["test_input"]
for test_question in tqdm(test_questions):
    # Finetune generation
    response = client.chat.completions.create(model="", # need to put the finetuned model here 
                                            messages=test_question, temperature=0.0, top_p=1.0, max_tokens=350)
    # # SB generation
    # response = client.chat.completions.create(model="", # need to put the finetuned model here 
    #                                         messages=test_question, temperature=1.0, top_p=1.0, max_tokens=10, n=20)    
    for choice in response.choices:
        responses.append(choice.message.content)

with open("gpt_responses/gpt_finetune.json", "w") as outfile:
    json.dump(responses, outfile, ensure_ascii=False, indent = 2)
    
# with open("gpt_responses/SB_sampling.json", "w") as outfile:
#     json.dump(responses, outfile, ensure_ascii=False, indent = 2)




