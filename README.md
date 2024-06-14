# Cheese Classification challenge

## Installation

Cloning the repo:
```
git clone git@github.com:Aymeric782/Cheese_classification_challenge.git
cd cheese_classification_challenge
```
Install dependencies:
```
conda create -n cheese_challenge python=3.10
conda activate cheese_challenge
pip install -r requirements.txt
```
Install dependencies for the OCR:
```
pip install fuzzywuzzy
pip install azure-cognitiveservices-vision-computervision msrest spacy
python -m spacy download en_core_web_md
```
Install dependencies for dreambooth:
```
cd peft/examples/lora_dreambooth
conda create -n peft python=3.10
conda activate peft
pip install -r requirements.txt
pip install git+https://github.com/huggingface/peft
```

Download the data from kaggle and copy them in the dataset folder
The data should be organized as follow: ```dataset/val```, ```dataset/test```.

## Using this codebase

### Fine-tuning

Go in the folder ```peft/examples/lora_dreambooth```:
```
cd peft/examples/lora_dreambooth
```

To fine-tune Stable Diffusion with DreamBooth on each cheese you can run
```
python exec_train.py
```
The models will be saved in the folder ```dreambooth_models```.

### Generating images

To generate cheese images with those models you can do 
```
python exec_prompt.py
```
The prompts used come from the folder ```Prompts```.

They were generated using the script ```Generation_prompts_caracteristiques.py``` in the same folder.

The images are saved in the folder ```classifier_dataset```.

### Training

Go back in the folder ```cheese_classification_challenge```.

To train your model you can run 

```
python train.py
```

This will save a checkpoint in checkpoints with the name of the experiment you have. Careful, if you use the same exp name it will get overwritten

to change experiment name, you can do

```
python train.py experiment_name=new_experiment_name
```

## Create submition

To create a submission file with the OCR, you can run 
```
python create_submition_OCR.py
```

If you want to create a submission file without the OCR, you can run 
```
python create_submition_OCR.py experiment_name="name_of_the_exp_you_want_to_score" model=config_of_the_exp
```
Make sure to specify the name of the checkpoint you want to score and to have the right model config