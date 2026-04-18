# Anime Sketch Colorization — Pix2Pix Conditional GAN

A project that colorizes anime line sketches using a **Pix2Pix Conditional GAN** trained on the Anime Sketch Colorization Pair dataset.

---

## Demo

🤗 **Live Demo:** [HuggingFace Space](https://huggingface.co/spaces/Sumit-Jethani/Sketch-colorization-pix2pix-GAN)

---

## Example Results

| Input Sketch | Colorized Output |
|---|---|
| Anime line sketch | Generated color image |

---

## Model Architecture

- **Generator:** U-Net with skip connections (Encoder-Decoder)
- **Discriminator:** PatchGAN (16×16 patch-level classification)
- **Loss:** Adversarial Loss + L1 Reconstruction Loss (λ=100)

---

## Training Details

| Parameter | Value |
|---|---|
| Dataset | Anime Sketch Colorization Pair |
| Image Size | 128×128 |
| Batch Size | 16 |
| Optimizer | Adam (lr=0.0002, β=0.5, 0.999) |
| Epochs | 35 (Early stopping at 20) |
| Best Epoch | 13 |
| GPUs | Dual T4 (Kaggle) |
| Mixed Precision | AMP (torch.cuda.amp) |

---

## Regularization

- Dropout (0.5) in Generator bottleneck and decoder
- Instance Normalization throughout
- L2 Weight Decay (1e-5) on both Generator and Discriminator

---

## Evaluation

| Metric | Score | Std Dev |
|---|---|---|
| SSIM | 0.8094 | ± 0.0745 |
| PSNR | 21.02 dB | ± 2.02 dB |

---

## Download Model

Download the pretrained model weights:

```
https://huggingface.co/spaces/Sumit-Jethani/Sketch-colorization-pix2pix-GAN/resolve/main/best_model.pt
```

Or via terminal:
```bash
wget https://huggingface.co/spaces/Sumit-Jethani/Sketch-colorization-pix2pix-GAN/resolve/main/best_model.pt
```

---

## Project Structure

```
├── app.py                              # Gradio app for HuggingFace deployment
├── requirements.txt                    # Dependencies
├── best_model.pt                       # Pretrained model weights
└── pix2pix-sketch-to-photo.ipynb       # Training notebook
```

---

## Installation & Usage

```bash
# Clone the repo
git clone https://github.com/sumitjethani/anime-sketch-colorization.git
cd anime-sketch-colorization

# Install dependencies
pip install -r requirements.txt

# Download model
wget https://huggingface.co/spaces/Sumit-Jethani/Sketch-colorization-pix2pix-GAN/resolve/main/best_model.pt

# Run the app
python app.py
```

---

## Requirements

```
torch
torchvision
gradio
Pillow
numpy
```

---

## Dataset

[Anime Sketch Colorization Pair](https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair) — Kaggle

---

## Author

**Sumit Jethani**
- GitHub: [github.com/sumitjethani](https://github.com/sumitjethani)
- LinkedIn: [linkedin.com/in/sumit-jethani](https://linkedin.com/in/sumit-jethani)
- HuggingFace: [huggingface.co/Sumit-Jethani](https://huggingface.co/Sumit-Jethani)
