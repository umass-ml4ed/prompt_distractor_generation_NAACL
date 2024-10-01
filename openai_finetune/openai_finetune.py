import os
from openai import OpenAI
import json
from tqdm import tqdm
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--sb", action="store_true")
args = parser.parse_args()

if args.sb:
    train_data_filename = "data/train_cor_data.jsonl"
    val_data_filename = "data/val_cor_data.jsonl"
    test_data_filename = "data/test_cor_data.json"
    output_filename = "SB_sampling_preproc.json"
else:
    train_data_filename = "data/train_data.jsonl"
    val_data_filename = "data/val_data.jsonl"
    test_data_filename = "data/test_data.json"
    output_filename = "gpt_finetune.json"

api_keys = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key = api_keys)
train_file = client.files.create(
    file=open(train_data_filename, "rb"),
    purpose="fine-tune"
)
train_file_id = train_file.id
val_file = client.files.create(
    file=open(val_data_filename, "rb"),
    purpose="fine-tune"
    )
val_file_id = val_file.id

client.fine_tuning.jobs.create(
    training_file=train_file_id,
    validation_file=val_file_id,
    model="gpt-3.5-turbo-1106",
    hyperparameters={"batch_size": 8, "n_epochs": 2})
jobs = client.fine_tuning.jobs.list(limit=10)
cur_job_id = next(job for job in jobs.data if job.training_file == train_file_id).id
print("Job ID", cur_job_id)

# Poll job status until training is done
max_time = 3600
sleep_time = 10
report_every = 6
cur_time = 0
while cur_time < max_time:
    cur_job = client.fine_tuning.jobs.retrieve(cur_job_id)
    if (cur_time // sleep_time) % report_every == 0:
        print("Time:", cur_time)
        print("Status:", cur_job.status)
        if cur_job.status == "running":
            print(client.fine_tuning.jobs.list_events(id=cur_job_id, limit=1).data[0].message)
    if cur_job.status == "succeeded":
        break
    time.sleep(sleep_time)
    cur_time += sleep_time

fine_tuned_model_id = cur_job.fine_tuned_model
print("Done! Fine-tuned Model ID:", fine_tuned_model_id)

# run on the test data, this code can be used for both FT and SB
responses = []
with open(test_data_filename, "r") as f:
    test_data = json.load(f)
    test_questions = test_data["test_input"]
for test_question in tqdm(test_questions):
    if args.sb:
        response = client.chat.completions.create(model=fine_tuned_model_id, messages=test_question, temperature=1.0, top_p=1.0, max_tokens=20, n=20) 
    else:
        response = client.chat.completions.create(model=fine_tuned_model_id, messages=test_question, temperature=0.0, top_p=1.0, max_tokens=350)

    for choice in response.choices:
        responses.append(choice.message.content)

with open(output_filename, "w") as outfile:
    json.dump(responses, outfile, ensure_ascii=False, indent = 2)
