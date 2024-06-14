# -*- coding: utf-8 -*-
"""
Created on Mon May 20 18:37:49 2024

@author: Loic et Sophie
"""

import os
import torch
import random as rd

from diffusers import StableDiffusionPipeline
from peft import PeftModel, LoraConfig

import subprocess
import os



folder = os.getcwd()

elt_dir = folder + "/dreambooth_dataset/"
model_dir = folder + "/dreambooth_models/"

os.environ['MODEL_NAME'] = "CompVis/stable-diffusion-v1-4"
os.environ['CLASS_DIR'] = elt_dir + "/class"



with open("list_of_cheese.txt", "r") as f:
    labels = f.readlines()
    labels = [label.strip() for label in labels]



for label in labels:

    os.environ['INSTANCE_DIR'] = elt_dir + label 
    os.environ['OUTPUT_DIR'] = model_dir + label

    command = [
        'accelerate', 'launch', 'train_dreambooth.py',
        '--pretrained_model_name_or_path', os.getenv('MODEL_NAME'),
        '--instance_data_dir', os.getenv('INSTANCE_DIR'),
        '--class_data_dir', os.getenv('CLASS_DIR'),
        '--output_dir', os.getenv('OUTPUT_DIR'),
        '--train_text_encoder',
        '--with_prior_preservation', '--prior_loss_weight=1.0',
        '--num_dataloader_workers=1',
        '--instance_prompt=a photo of ' + label + ' cheese',
        '--class_prompt=a photo of cheese',
        '--resolution=512',
        '--train_batch_size=1',
        '--lr_scheduler=constant',
        '--lr_warmup_steps=0',
        '--num_class_images=200',
        '--use_lora',
        '--lora_r=16',
        '--lora_alpha=27',
        '--lora_text_encoder_r=16',
        '--lora_text_encoder_alpha=17',
        '--learning_rate=1e-4',
        '--gradient_accumulation_steps=1',
        '--gradient_checkpointing',
        '--max_train_steps=800'
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    # Affichage des erreurs
    print(result.stdout)
    print(result.stderr)


