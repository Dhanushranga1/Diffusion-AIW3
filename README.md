# Diffusion-AIW3
# 1. Diffusion-Based Image Denoising (CIFAR-10)



---

## Objective

Train a denoising model using diffusion methods (starting from a basic U-Net and evolving toward DDPM-style noise learning). Evaluate quality using standard metrics like PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index).

The final version shared here gives the best denoising results out of all attempts so far.

---

## Learning Resources

I learned the concepts from the paper *Denoising Diffusion Probabilistic Models (DDPM)* and also followed the video "**The Diffusion Model’s Unsung Sidekick**" from the YouTube channel **Depth First**, which explains the noise scheduling and timestep conditioning logic in a clear way.

---

## Final Model Performance (v1 – Best Result)

- **Dataset**: CIFAR-10
- **Model**: Simplified UNet
- **Training**: 15 epochs
- **Noise**: Gaussian noise with σ = 0.2
- **Loss Function**: MSE (pixelwise)
- **Metrics**:
  - Noisy PSNR: 14.64 dB → Denoised PSNR: **25.11 dB**
  - Noisy SSIM: 0.400 → Denoised SSIM: **0.840**

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

## Key Observations

- A deeper U-Net trained for more epochs gives better results than lightly trained DDPMs on small datasets.
- SSIM improves significantly with longer training and proper noise control.
- DDPM-based models are more robust theoretically but require careful time conditioning and longer training to beat traditional setups.

---

## Next Steps

The next part of this task involves training a **Stable Diffusion model** for **text-to-image generation**. Once that is complete, I’ll update this repository with a second notebook, generated images, and benchmark discussions.


