# [Exploring Automated Distractor Generation for Math Multiple-choice Questions via Large Language Models](https://arxiv.org/abs/2404.02124)

In this repository, we present the code to our paper "Exploring Automated Distractor Generation for Math Multiple-choice Questions via Large Language Models" by Wanyong Feng, Jaewook Lee, Hunter McNichols, Alexander Scarlatos, Digory Smith, Simon Woodhead, Nancy Otero Ornelas, and Andrew Lan. We explore a variety of approaches to this task, including in-context learning(kNN), fine-tuning(FT), and chain-of-thought prompting(CoT), together with rule(RB)- and sampling-based(SB) baselines. The paper is accepted as the findings of NAACL 2024.

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
### Generate Distractors with FT approach (GPT3.5)
```
python openai_finetune.py (in openai_finetune folder)
python run.py 'command.task=fetch_from_openai' 'data.trainFilepath=data/<output_file>.csv' 'data.testFilepath=data/<output_file>.csv' 'openAI.model=gpt-4-1106-preview' 'prompt.type=zero_shot' 'retriever.type=none' 'retriever.encodingPattern=q+a+f' 'dir_finetune_result.model_name=gpt_finetune'
```
### Generate Distractors with FT approach (Mistral)
```
python train.py (in mistral_finetune folder)
python test.py (in mistral_finetune folder)
python run.py 'command.task=fetch_from_openai' 'data.trainFilepath=data/<output_file>.csv' 'data.testFilepath=data/<output_file>.csv' 'openAI.model=gpt-4-1106-preview' 'prompt.type=zero_shot' 'retriever.type=none' 'retriever.encodingPattern=q+a+f' 'dir_finetune_result.model_name=mistral_finetune'
```
### Generate Distractors with SB approach
```
python openai_finetune.py
python run.py 'command.task=fetch_from_openai' 'data.trainFilepath=data/<output_file>.csv' 'data.testFilepath=data/<output_file>.csv' 'openAI.model=gpt-4-1106-preview' 'prompt.type=zero_shot' 'retriever.type=none' 'retriever.encodingPattern=q+a+f' 'dir_finetune_result.model_name=SB_sampling'
### Evaluating Distractors
```
python evaluation.py
```

## Citation
If you used our code or found this work useful in any way, please cite us!
```
@misc{feng2024exploring,
      title={Exploring Automated Distractor Generation for Math Multiple-choice Questions via Large Language Models}, 
      author={Wanyong Feng and Jaewook Lee and Hunter McNichols and Alexander Scarlatos and Digory Smith and Simon Woodhead and Nancy Otero Ornelas and Andrew Lan},
      year={2024},
      eprint={2404.02124},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```  
