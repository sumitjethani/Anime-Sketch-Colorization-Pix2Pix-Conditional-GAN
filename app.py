import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import gradio as gr

# ─── Model Architecture ───────────────────────────────────────────────────────

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, act='relu', dropout=0.0):
        super().__init__()
        layers = []
        if down:
            layers += [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        else:
            layers += [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        layers += [nn.InstanceNorm2d(out_ch)]
        layers += [nn.ReLU(True) if act == 'relu' else nn.LeakyReLU(0.2, True)]
        if dropout > 0:
            layers += [nn.Dropout(dropout)]
        self.block = nn.Sequential(*layers)

    def forward(self, x): return self.block(x)


class Generator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dropout=0.5):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(in_ch, 64, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.e2 = UNetBlock(64,  128, down=True,  act='lrelu')
        self.e3 = UNetBlock(128, 256, down=True,  act='lrelu')
        self.e4 = UNetBlock(256, 512, down=True,  act='lrelu')
        self.e5 = UNetBlock(512, 512, down=True,  act='lrelu')
        self.e6 = UNetBlock(512, 512, down=True,  act='lrelu', dropout=dropout)
        self.bottleneck = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU(True))
        self.d1 = UNetBlock(512,  512, down=False, dropout=dropout)
        self.d2 = UNetBlock(1024, 512, down=False, dropout=dropout)
        self.d3 = UNetBlock(1024, 512, down=False)
        self.d4 = UNetBlock(1024, 256, down=False)
        self.d5 = UNetBlock(512,  128, down=False)
        self.d6 = UNetBlock(256,  64,  down=False)
        self.out = nn.Sequential(nn.ConvTranspose2d(128, out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        b  = self.bottleneck(e6)
        d1 = self.d1(b)
        d2 = self.d2(torch.cat([d1, e6], 1))
        d3 = self.d3(torch.cat([d2, e5], 1))
        d4 = self.d4(torch.cat([d3, e4], 1))
        d5 = self.d5(torch.cat([d4, e3], 1))
        d6 = self.d6(torch.cat([d5, e2], 1))
        return self.out(torch.cat([d6, e1], 1))


# ─── Load Model ───────────────────────────────────────────────────────────────

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ckpt = torch.load('best_model.pt', map_location=DEVICE)
G = Generator(dropout=0.5).to(DEVICE)

# handle DataParallel checkpoint keys
state = ckpt['G_state']
if any(k.startswith('module.') for k in state.keys()):
    state = {k.replace('module.', ''): v for k, v in state.items()}

G.load_state_dict(state)
G.eval()
print('Model loaded.')

# ─── Inference ────────────────────────────────────────────────────────────────

transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])

def colorize(sketch_img):
    img = sketch_img.convert('RGB')
    original_size = img.size  # save original size (w, h)
    inp = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = G(inp)

    out = (out.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
    out = out.cpu().permute(1, 2, 0).numpy()
    out = (out * 255).astype(np.uint8)
    result = Image.fromarray(out)
    result = result.resize(original_size, Image.LANCZOS)  # resize back to input size
    return result


# ─── Gradio UI ────────────────────────────────────────────────────────────────

demo = gr.Interface(
    fn=colorize,
    inputs=gr.Image(type='pil', label='Input Sketch'),
    outputs=gr.Image(type='pil', label='Colorized Output'),
    title='Anime Sketch Colorization — Pix2Pix GAN',
    examples=[],
    theme=gr.themes.Soft(),
)

if __name__ == '__main__':
    demo.launch()