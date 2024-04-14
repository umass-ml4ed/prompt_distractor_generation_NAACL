## Prompt-based distractor gen project

## Installation

### Conda (reccomended)
`conda env create --file enviornment.yml`

### Pip
`pip install -r requirements.txt`

### Directories
```
mkdir data
mkdir analysis
mkdir cache
```

## Run

### Generate Distractors
```
python run.py 'command.task=fetch_from_openai' 'data.trainFilepath=analysis/<output_file>.csv' 'openAI.model=gpt-3.5-turbo-1106'
```

## Acknowledgement
we use GitHub Copilot to help us write some data-processing and utility functions  