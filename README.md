# [Exploring Automated Distractor Generation for Math Multiple-choice Questions via Large Language Models](https://aclanthology.org/2024.findings-naacl.193/)

In this repository, we present the code to our paper "Exploring Automated Distractor Generation for Math Multiple-choice Questions via Large Language Models" by Wanyong Feng, Jaewook Lee, Hunter McNichols, Alexander Scarlatos, Digory Smith, Simon Woodhead, Nancy Otero Ornelas, and Andrew Lan. We explore a variety of approaches to distractor generation, including in-context learning(kNN), fine-tuning(FT), and chain-of-thought prompting(CoT), together with rule(RB)- and sampling-based(SB) baselines. The paper is accepted at the findings of NAACL 2024.

For any questions please [email](mailto:wanyongfeng@umass.edu) or raise an issue.

## Installation

### Conda (reccomended)
`conda env create --file enviornment.yml`

### Pip
`pip install -r requirements.txt`

## Run

### Generate Distractors with kNN approach
```
python run.py 'command.task=fetch_from_openai' 'data.trainFilepath=data/<output_file>.csv' 'data.testFilepath=data/<output_file>.csv' 'openAI.model=gpt-3.5-turbo-1106' 'prompt.type=distractor_and_answer_with_feedback' 'retriever.type=KNN' 'retriever.encodingPattern=q+a+f'
```
### Generate Distractors with CoT approach
```
python run.py 'command.task=fetch_from_openai' 'data.trainFilepath=data/<output_file>.csv' 'data.testFilepath=data/<output_file>.csv' 'openAI.model=gpt-4-1106-preview' 'prompt.type=zero_shot' 'retriever.type=none' 'retriever.encodingPattern=q+a+f'
```
### Generate Distractors with RB approach
```
python misconception_selection.py
python run.py 'command.task=fetch_from_openai' 'data.trainFilepath=data/<output_file>.csv' 'data.testFilepath=data/<output_file>.csv' 'openAI.model=gpt-4-1106-preview' 'prompt.type=rule_based_selection' 'retriever.type=misconception_selection' 'retriever.encodingPattern=q+a+f'
```
### Generate Distractors with FT approach (GPT3.5, in openai_finetune folder)
```
python openai_finetune.py
python run.py 'command.task=fetch_from_openai' 'data.trainFilepath=data/<output_file>.csv' 'data.testFilepath=data/<output_file>.csv' 'openAI.model=gpt-4-1106-preview' 'prompt.type=zero_shot' 'retriever.type=none' 'retriever.encodingPattern=q+a+f' 'dir_finetune_result.model_name=gpt_finetune'
```
### Generate Distractors with FT approach (Mistral, in mistral_finetune folder)
```
python train.py
python test.py
python run.py 'command.task=fetch_from_openai' 'data.trainFilepath=data/<output_file>.csv' 'data.testFilepath=data/<output_file>.csv' 'openAI.model=gpt-4-1106-preview' 'prompt.type=zero_shot' 'retriever.type=none' 'retriever.encodingPattern=q+a+f' 'dir_finetune_result.model_name=mistral_finetune'
```
### Generate Distractors with SB approach
```
python openai_finetune.py
python run.py 'command.task=fetch_from_openai' 'data.trainFilepath=data/<output_file>.csv' 'data.testFilepath=data/<output_file>.csv' 'openAI.model=gpt-4-1106-preview' 'prompt.type=zero_shot' 'retriever.type=none' 'retriever.encodingPattern=q+a+f' 'dir_finetune_result.model_name=SB_sampling'
```
### Evaluating Distractors
```
python evaluation.py
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
