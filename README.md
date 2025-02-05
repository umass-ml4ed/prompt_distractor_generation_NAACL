# [Exploring Automated Distractor Generation for Math Multiple-choice Questions via Large Language Models](https://aclanthology.org/2024.findings-naacl.193/)

In this repository, we present the code to our paper "Exploring Automated Distractor Generation for Math Multiple-choice Questions via Large Language Models" by Wanyong Feng, Jaewook Lee, Hunter McNichols, Alexander Scarlatos, Digory Smith, Simon Woodhead, Nancy Otero Ornelas, and Andrew Lan. We explore a variety of approaches to distractor generation, including in-context learning(kNN), fine-tuning(FT), and chain-of-thought prompting(CoT), together with rule(RB)- and sampling-based(SB) baselines. The paper is accepted at the findings of NAACL 2024.

For any questions please [email](mailto:wanyongfeng@umass.edu) or raise an issue.

## Installation

### Pip
`pip install -r requirements.txt`

### Conda
`conda env create --file enviornment.yml`

## Data Format

The data we use in these experiments is private. However, if you have your own data you want to test on, you can format it as follows in .csv files:

- `id`: the unique question ID
- `question`: the question text
- `correct_option`: a JSON object containing:
  - `option`: the correct option text
  - `explanation`: the textual solution for the question
- `construct_info`: a JSON object containing:
  - `construct1`: a list for the top-level construct: `[construct ID: int, construct description: str]`
  - `construct2`: a list for the mid-level construct: `[construct ID: int, construct description: str]`
  - `construct3`: a list for the low-level construct: `[construct ID: int, construct description: str]`
- `distractors`: a JSON list of 3 objects (one for each distractor), each containing:
  - `option_idx`: the index of the option in the question (1-4)
  - `option`: the distractor option text
  - `explanation`: the feedback/explanation text for the distractor
  - `proportion`: the percentage of students that chose this option (set to 0 if unknown)
  - `misconception`: the text for the misconception for this distractor (optional, only needed for misconception-based ICL prompts or rule-based method)

You can specify the train and test file paths using the `data.trainFilepath` and `data.testFilepath` command line arguments.

## Run

### Generate Distractors with kNN approach
```
python run.py 'openAI.model=gpt-3.5-turbo-1106' 'prompt.type=distractor_and_answer_with_feedback' 'retriever.type=KNN' 'retriever.encodingPattern=q+a+f'
```
### Generate Distractors with CoT approach
```
python run.py 'openAI.model=gpt-4-1106-preview' 'prompt.type=zero_shot' 'retriever.type=none'
```
### Generate Distractors with RB approach
```
python misconception_selection.py
python run.py 'openAI.model=gpt-4-1106-preview' 'prompt.type=rule_based_selection' 'retriever.type=misconception_selection' 'retriever.encodingPattern=q+a+f' 'data.testFilepath=data/eedi_test_20_cleaned_4_18_misconceptions.csv'
```
### Generate Distractors with FT approach (GPT3.5)
```
python openai_finetune/data_processing.py
python openai_finetune/openai_finetune.py
python run.py 'dir_finetune_result.model_name=gpt_finetune' 'prompt.type=zero_shot' 'retriever.type=none'
```
### Generate Distractors with FT approach (Mistral)
```
python mistral_finetune/data_processing.py
python mistral_finetune/train.py
python mistral_finetune/test.py --model_checkpoint <model_checkpoint_folder_path>
python run.py 'dir_finetune_result.model_name=mistral_finetune' 'prompt.type=zero_shot' 'retriever.type=none'
```
### Generate Distractors with SB approach (GPT3.5)
```
python openai_finetune/sb_data_processing.py
python openai_finetune/openai_finetune.py --sb
python openai_finetune/post_sb_processing.py
python run.py 'dir_finetune_result.model_name=SB_sampling' 'prompt.type=zero_shot' 'retriever.type=none'
```
### Generate Distractors with SB approach (Mistral)
```
python mistral_finetune/sb_data_processing.py
python mistral_finetune/train.py --sb
python mistral_finetune/test.py --sb --model_checkpoint <model_checkpoint_folder_path>
python mistral_finetune/post_sb_processing.py <num_distractors>
python run.py 'dir_finetune_result.model_name=mistral_SB' 'prompt.type=zero_shot' 'retriever.type=none'
```
### Evaluating Distractors
```
python evaluation.py analysis/<result_filename> <num_distractors>
```

## Citation
If you used our code or found this work useful in any way, please cite us!
```
@inproceedings{feng-etal-2024-exploring,
    title = "Exploring Automated Distractor Generation for Math Multiple-choice Questions via Large Language Models",
    author = "Feng, Wanyong  and
      Lee, Jaewook  and
      McNichols, Hunter  and
      Scarlatos, Alexander  and
      Smith, Digory  and
      Woodhead, Simon  and
      Ornelas, Nancy  and
      Lan, Andrew",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2024",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-naacl.193",
    pages = "3067--3082",
}
```  
