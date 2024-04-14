## Prompt-based Math MCQ Distractor Generation

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

## Acknowledgement
we use GitHub Copilot to help us write some data-processing and utility functions  
