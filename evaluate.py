import os
import torch
from diffusers import FluxPipeline
import time
import glob
from PIL import Image
from evaluate import calculate_clip_score, calculate_fid_score
import torchvision.transforms as transforms
import numpy as np
import json
import math

PRE_TRAINED_MODEl = "black-forest-labs/FLUX.1-dev"
FINE_TUNED_MODEL = "/home/apple/ai-toolkit/output/fashion-generation-h-and-m-V1.0/fashion-generation-h-and-m-V1.0.safetensors"
VAL_IMAGES_DIR = "/home/apple/fashion_captions_3500/"
VAL_RESULTS_DIR = "/home/apple/eval_results/"

num_images = 250

# Load Flux
pipe = FluxPipeline.from_pretrained(PRE_TRAINED_MODEl, torch_dtype=torch.float16).to("cuda")

# Load your fine-tuned model
pipe.load_lora_weights(FINE_TUNED_MODEL, adapter_name="default")


def generate_images(prompts):
    images = pipe(prompts, height=512, width=512, num_images_per_prompt=1, output_type="np").images

    return images

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

if __name__ == '__main__':
    real_images = []
    all_prompts = []
    fake_images = []
    
    # create eval directory
    os.makedirs(VAL_RESULTS_DIR, exist_ok=True)
        
    if num_images > 10:
        splits = list(split(range(1, num_images+1), math.ceil(num_images/10)))
    else:
        splits = [range(1, num_images+1)]

    print(f"splits:{splits}")

    for index0, split0 in enumerate(splits):
        
        prompts = []
        print(f"processing split:{split0}")
        
        for index in split0:
            with open(os.path.join(VAL_IMAGES_DIR, f"fashion_image_{index}.txt"), "r") as f:
                prompt = f.readline()
                prompts.append(prompt)
        
        # Generate an image
        images = generate_images(prompts)
        
        # save images
        for index1, image in enumerate(images):
            # save image
            img = Image.fromarray((image*255).astype('uint8'), 'RGB')
            img.save(os.path.join(VAL_RESULTS_DIR, f"gen_image_{split0[index1]}.jpg"))

            # add to fake_images
            fake_images.append(np.array(img))
            
            image1 = Image.open(os.path.join(VAL_IMAGES_DIR, f"fashion_image_{split0[index1]}.jpg"))
            image1.save(os.path.join(VAL_RESULTS_DIR, f"real_image_{split0[index1]}.jpg"))

            # all to all real images
            real_images.append(image1)

            # save prompt
            with open(os.path.join(VAL_RESULTS_DIR, f"gen_image_{split0[index1]}.txt"), "w") as f:
                with open(os.path.join(VAL_IMAGES_DIR, f"fashion_image_{split0[index1]}.txt"), "r") as f1:
                    f.write(f1.read())
                    all_prompts.append(f1.read())

    real_images = np.stack(real_images, axis=0)
    fake_images = np.stack(fake_images, axis=0)
    
    print(real_images, fake_images, all_prompts)

    clip_score = calculate_clip_score(real_images, all_prompts)
    print(f"CLIP score: {clip_score}")

    fake_images = torch.tensor(fake_images)
    fake_images = fake_images.permute(0, 3, 1, 2)
    
    real_images = torch.tensor(real_images)
    real_images = real_images.permute(0, 3, 1, 2)

    fid_score = calculate_fid_score(real_images, fake_images)
    print(f"FID score: {fid_score}")

    with open(os.path.join(VAL_RESULTS_DIR, f"score.json"), "w") as f:
        json.dump({"FID": fid_score, "CLIP": clip_score}, f)
    
    # for index1, prompt in enumerate(all_prompts):
    #     # save image
    #     img = Image.fromarray((image*255).astype('uint8'), 'RGB')
    #     img.save(os.path.join(VAL_RESULTS_DIR, f"gen_image_{index1+1}.jpg"))

    #     image1 = Image.open(os.path.join(VAL_IMAGES_DIR, f"fashion_image_{index1+1}.jpg"))
    #     image1.save(os.path.join(VAL_RESULTS_DIR, f"real_image_{index1+1}.jpg"))
        
    #     # save prompt
    #     with open(os.path.join(VAL_RESULTS_DIR, f"gen_image_{index1+1}.txt"), "w") as f:
    #         f.write(prompts[index1])
            
    # print(images, prompts)

    # print(images, images.shape)

    # from diffusers import StableDiffusionPipeline
    # import torch
    
    # model_ckpt = "CompVis/stable-diffusion-v1-4"
    # sd_pipeline = StableDiffusionPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16).to("cuda")

    # prompts = [
    #     "a photo of an astronaut riding a horse on mars",
    #     "A high tech solarpunk utopia in the Amazon rainforest",
    #     "A pikachu fine dining with a view to the Eiffel Tower",
    #     "A mecha robot in a favela in expressionist style",
    #     "an insect robot preparing a delicious meal",
    #     "A small cabin on top of a snowy mountain in the style of Disney, artstation",
    # ]
    
    # images = sd_pipeline(prompts, num_images_per_prompt=1, output_type="np").images

    # print(images.shape, type(images))
    
    # # Generate an image
    # prompt = "a fitted black party dress with a high waistline, made of medium-stretch satin and lined with 100% polyester, featuring a notch neckline, sleeveless design, mermaid hemline, and a zipper detail, all in a solid pattern."
    
    # image = generate_image(prompt)
    
    # os.makedirs("results", exist_ok=True)
    
    # # Save the image
    # image.save(f"results/fashion_{int(time.time())}.png")

    
    