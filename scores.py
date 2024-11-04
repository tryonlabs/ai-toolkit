from torchmetrics.functional.multimodal import clip_score
from functools import partial
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

def calculate_fid_score(real_images, fake_images):
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    return float(fid.compute())
    

if __name__ == '__main__':
    sd_clip_score = calculate_clip_score(images, prompts)
    print(f"CLIP score: {sd_clip_score}")