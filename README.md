# [Exploring Automated Distractor Generation for Math Multiple-choice Questions via Large Language Models](https://arxiv.org/abs/2404.02124)

In this repository, we present the code to our paper "Exploring Automated Distractor Generation for Math Multiple-choice Questions via Large Language Models" by Wanyong Feng, Jaewook Lee, Hunter McNichols, Alexander Scarlatos, Digory Smith, Simon Woodhead, Nancy Otero Ornelas, and Andrew Lan. We explore a variety of approaches to this task, including in-context learning(kNN), fine-tuning(FT), and chain-of-thought prompting(CoT), together with rule(RB)- and sampling-based(SB) baselines. The paper is accepted as the findings of NAACL 2024.

For any questions please [email](mailto:wanyongfeng@umass.edu) or raise an issue.

## Installation

### Conda (reccomended)
`conda env create --file enviornment.yml`

### Pip
`pip install -r requirements.txt`

## Run

### Generate Distractors
```
python run.py 'command.task=fetch_from_openai' 'data.trainFilepath=data/<output_file>.csv' 'openAI.model=gpt-3.5-turbo-1106'
```
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
