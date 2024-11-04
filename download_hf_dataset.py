import os
from datasets import load_dataset

DATA_DIR = "/home/apple/h-and-m-fashion-caption-12k"
os.makedirs(DATA_DIR, exist_ok=True)

ds = load_dataset("tomytjandra/h-and-m-fashion-caption-12k")

for index, data in enumerate(ds['train']):
    print(index, data['text'], data['image'])

    data['image'].save(os.path.join(DATA_DIR, f"img_cap_{index+1}.jpg"))
    with open(os.path.join(DATA_DIR, f"img_cap_{index+1}.txt"), "w") as f:
        f.write(data['text'])
