from diffusers import FluxPipeline
import torch


PRE_TRAINED_MODEl = "black-forest-labs/FLUX.1-dev"
FINE_TUNED_MODEL = "/home/apple/ai-toolkit/output/fashion-generation-h-and-m-V1.0/fashion-generation-h-and-m-V1.0.safetensors"


# Load Flux
pipe = FluxPipeline.from_pretrained(PRE_TRAINED_MODEl, torch_dtype=torch.float16).to("cuda")

# Load your fine-tuned model
pipe.load_lora_weights(FINE_TUNED_MODEL, adapter_name="default")

# Push to the hub
pipe.push_to_hub("tryonlabs/FLUX.1-dev-LoRA-Outfit-Generator")
