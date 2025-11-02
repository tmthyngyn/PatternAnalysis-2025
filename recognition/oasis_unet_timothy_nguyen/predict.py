import torch
import matplotlib.pyplot as plt
from modules import UNet2D
from dataset import OASIS2DSegmentation

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2D(in_channels=1, n_classes=4).to(device)
    model.load_state_dict(torch.load("checkpoints/oasis_unet.pth", map_location=device))
    model.eval()

    ds = OASIS2DSegmentation(split="train")
    img, mask = ds[0]
    with torch.no_grad():
        logits = model(img.unsqueeze(0).to(device))
        pred = logits.argmax(1).squeeze(0).cpu()

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].imshow(img.squeeze(0), cmap="gray"); axs[0].set_title("image")
    axs[1].imshow(mask, cmap="nipy_spectral"); axs[1].set_title("gt")
    axs[2].imshow(pred, cmap="nipy_spectral"); axs[2].set_title("pred")
    for ax in axs: ax.axis("off")
    plt.tight_layout()
    plt.savefig("prediction_example.png")

if __name__ == "__main__":
    main()
