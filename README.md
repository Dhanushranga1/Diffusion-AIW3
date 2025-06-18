# Diffusion-AIW3

## 1. Diffusion-Based Image Denoising (CIFAR-10)  
## 2. Text-to-Image Fine-Tuning using LoRA on Stable Diffusion

---

## Objective

1. **Image Denoising**: Train a noise-aware denoising model using diffusion-based methods, evolving from a basic U-Net to a DDPM-style setup. Evaluate results using PSNR and SSIM metrics.
2. **Text-to-Image Fine-Tuning**: Fine-tune a pre-trained Stable Diffusion model (`runwayml/stable-diffusion-v1-5`) using a small custom dataset of car images with **LoRA** (Low-Rank Adaptation). The goal is to enable the model to generate better car-related images while keeping the training lightweight and efficient.

---

## Learning Resources

- For image denoising:  
  - **Paper**: *Denoising Diffusion Probabilistic Models (DDPM)*  
  - **Video**: [The Diffusion Model’s Unsung Sidekick](https://youtu.be/Fk2I6pa6UeA?si=yLBIA5zo1oqtdG8q) – by Depth First (explains timesteps and noise scheduling clearly)

- For LoRA + Stable Diffusion:
  - HuggingFace `diffusers` docs  
  - PEFT library for injecting and training LoRA adapters  
  - Community notebooks and minimal image datasets for reference

---

## Final Model Performance

###  Image Denoising (CIFAR-10)
- **Best Version**: v1 (trained longer on the same U-Net)
- **Metrics**:
  - **Noisy PSNR**: 14.64 dB → **Denoised PSNR**: 25.11 dB
  - **Noisy SSIM**: 0.400 → **Denoised SSIM**: 0.840
- **Takeaway**: Clean, sharp results using just pixel-level MSE and Gaussian noise.

---

### Stable Diffusion Fine-Tuning (LoRA on Car Images)
- **Base Model**: `runwayml/stable-diffusion-v1-5`  
- **LoRA Config**:
  - `r=16`, `lora_alpha=16`, `lora_dropout=0.1`
  - Target Modules: `["to_q", "to_k", "to_v", "to_out.0"]` in the UNet
- **Text Encoder**: `openai/clip-vit-large-patch14`  
- **Scheduler**: `DDIMScheduler`  
- **Optimizer**: AdamW  
- **Loss**: MSE between predicted and true noise (Diffusion-based loss)
- **Custom Dataset**: 9 car images, resized to 512×512  
- **Training Steps**: 400  
- **Output**: LoRA-adapted model saved at `./lora_car_model` and later merged for inference.

---

## Summary of Experiments

### Version 0 (v0)
| Component         | Description                                           |
|------------------|-------------------------------------------------------|
| Model            | Basic U-Net (5 epochs)                                |
| Loss             | MSE only                                              |
| Noise            | Additive Gaussian noise, σ = 0.2                      |
| Results          | PSNR: 24.59 dB, SSIM: 0.836                           |
| Notes            | Worked decently, blurry edges, low training time     |

---

### Version 1 (v1 – Final)
| Component         | Description                                           |
|------------------|-------------------------------------------------------|
| Model            | Same U-Net, trained for 15 epochs                     |
| Loss             | MSE                                                   |
| Noise            | Gaussian (σ = 0.2), clipped                           |
| Results          | PSNR: 25.11 dB, SSIM: 0.840                           |
| Notes            | Final version, best overall visual and metric scores |

---

### Version 2 (v2 – Experimental DDPM)

| Component         | Description                                           |
|------------------|-------------------------------------------------------|
| Model            | U-Net with timestep embedding                         |
| Loss             | MSE (predicting noise)                                |
| Noise            | DDPM noise formulation (sqrt(alpha_bar)*x + sqrt(1 - alpha_bar)*z) |
| Results          | PSNR: 16.16 dB, SSIM: 0.3757                          |
| Notes            | Promising approach but needs more epochs and tuning  |

---

### Stable Diffusion + LoRA Fine-Tuning (Cars)

| Component         | Description                                           |
|------------------|-------------------------------------------------------|
| Base Model        | `runwayml/stable-diffusion-v1-5`                      |
| Dataset           | 9 car images (512x512)                                |
| Scheduler         | DDIM                                                  |
| Tokenizer         | CLIP (ViT-L/14)                                       |
| Loss              | MSE between noise prediction and target               |
| Output            | LoRA weights saved at `./lora_car_model`             |
| Notes             | Required precise dtype matching (float16), proper latent handling with VAE, and merging LoRA after training for inference |

---

image generated post fine tuning Stable Diffusion + Lora Fine-Tuning
Prompt:
prompt = "a red sports car on a highway, cinematic lighting, ultra detailed"

![car_lora_sample](https://github.com/user-attachments/assets/fe550e81-6778-42d5-86bc-e780b4a5b9b1)


Images generated using the general Stable Diffusion model from hugging face 

Prompt 1: Prompt: "A cozy mountain cabin during snowfall, warm lights glowing inside"![sample_image_5](https://github.com/user-attachments/assets/29f7058b-b942-4744-af1c-bfde9a280d0d)

Prompt 2: Prompt: "An astronaut riding a horse on Mars, dramatic ligh![sample_image_4](https://github.com/user-attachments/assets/d719a780-4d7a-4bf6-8e66-e22a38659da1)
ting"
Prompt![sample_image_3](https://github.com/user-attachments/assets/531de65c-c7c6-4708-83b0-b83f8ca04ed5)
 3: Prompt: "A majestic lion with a glowing mane, fantasy setting"
Prompt![sample_image_2](https://github.com/user-attachments/assets/6194b668-3790-44e1-bf8c-73885f84c615)
 4: Prompt:![sample_image_2](https://github.com/user-attachments/assets/b5145730-71ab-43e5-96e7-2ee6039def0b)
 "A futuristic Tokyo street at night, neon lights and rain"


5: Prompt: ""![sample_image_1](https://github.com/user-attachments/assets/b1829518-e121-4202-974e-8262d335860c)
A surreal landscape with floating islands and waterfalls"

## Issues Faced (During LoRA Fine-Tuning)

- **Image shape mismatch**: Latents expected 4-channel input, but images were 3-channel RGB → fixed by using the VAE to convert images to latent space.
- **Dtype mismatch**: UNet with LoRA was in float16 while inputs were float32 → fixed by casting text embeddings and latents to `.half()`.
- **Loss.backward() error**: RuntimeError when trying to backward more than once → resolved by ensuring only one backward per loop and resetting optimizer.
- **Image quality**: Initial samples looked weird → verified resolution (512x512) and dataset cleaning step to resize all images.

---

## Inference Results

After training and merging LoRA weights back into UNet, we generated text-conditioned car images. Prompting worked well with both simple prompts ("a car on road") and styled prompts ("a futuristic car in desert").

---

## Next Steps

- Experiment with longer training or using DreamBooth-style datasets.
- Explore accelerating inference using ONNX or quantization.
- Try LoRA on other SD modules (e.g., attention in the text encoder).
- Deploy in a web app that allows users to upload 5–10 images and fine-tune SD using LoRA automatically.
