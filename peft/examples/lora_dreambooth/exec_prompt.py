import os
import torch
import random as rd

from diffusers import StableDiffusionPipeline
from peft import PeftModel, LoraConfig


n_img = 100   #nombre d'images à générer par type de fromage

folder = os.getcwd()
model_dir = folder + "/dreambooth_models/"
image_dir = folder + "/classifier_dataset/"

# context = [" on a plate", " on a platter", " on a display", " on a white background", ""]
# view = [" seen from above", " seen from afar", " seen up close", " seen from the side", ""]
# lc = len(context)
# lv = len(view)

def get_lora_sd_pipeline(
    ckpt_dir, base_model_name_or_path=None, dtype=torch.float16, device="cuda", adapter_name="default"
):
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    if os.path.exists(text_encoder_sub_dir) and base_model_name_or_path is None:
        config = LoraConfig.from_pretrained(text_encoder_sub_dir)
        base_model_name_or_path = config.base_model_name_or_path

    if base_model_name_or_path is None:
        raise ValueError("Please specify the base model name or path")

    pipe = StableDiffusionPipeline.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)

    if os.path.exists(text_encoder_sub_dir):
        pipe.text_encoder = PeftModel.from_pretrained(
            pipe.text_encoder, text_encoder_sub_dir, adapter_name=adapter_name
        )

    if dtype in (torch.float16, torch.bfloat16):
        pipe.unet.half()
        pipe.text_encoder.half()

    pipe.to(device)
    return pipe



with open("list_of_cheese.txt", "r") as f:
    labels = f.readlines()
    labels = [label.strip() for label in labels]



for label in labels:

    pipe = get_lora_sd_pipeline(model_dir + label, adapter_name="cheese")

    with open("Prompts/" + label + ".csv", "r") as f:
        prompting = f.readlines()
        prompting = [prompt.strip() for prompt in prompting]

    for k in range(n_img):

        # nc = rd.randint(0, lc-1)
        # nv = rd.randint(0, lv-1)

        # prompt = label + " cheese" + context[nc] + view[nv]

        prompt = prompting[k]
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]
        image.save(image_dir + label + "/" + str(k) + ".jpg")