from diffusers import FluxPipeline
import torch


PRE_TRAINED_MODEl = "black-forest-labs/FLUX.1-dev"
FINE_TUNED_MODEL = "/home/apple/ai-toolkit/output/fashion-generation-h-and-m-V1.0/fashion-generation-h-and-m-V1.0.safetensors"


# Load Flux
pipe = FluxPipeline.from_pretrained(PRE_TRAINED_MODEl, torch_dtype=torch.float16).to("cuda")

# Load your fine-tuned model
pipe.load_lora_weights(FINE_TUNED_MODEL, adapter_name="default")


def generate_images(prompts):
    images = pipe(prompts, height=512, width=512, num_images_per_prompt=1).images

    return images

images = generate_images(["solid black jersey top with narrow shoulder straps"])

images[0].save("generate_image.jpg")

