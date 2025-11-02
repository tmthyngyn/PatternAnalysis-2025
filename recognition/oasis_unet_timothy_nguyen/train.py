import os
import torch
from torch.utils.data import DataLoader, random_split
from dataset import OASIS2DSegmentation
from modules import UNet2D
from utils import one_hot_encode
import matplotlib.pyplot as plt

NUM_CLASSES = 4

def dice_score(logits, targets, num_classes=NUM_CLASSES):
    preds = torch.softmax(logits, dim=1)
    target_1h = one_hot_encode(targets, num_classes).to(logits.device)
    intersection = (preds * target_1h).sum(dim=(2,3))
    union = preds.sum(dim=(2,3)) + target_1h.sum(dim=(2,3))
    dice = (2. * intersection / (union + 1e-8)).mean()
    return dice

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = OASIS2DSegmentation(split="train")
    n_val = max(1, int(0.2 * len(ds)))
    n_train = len(ds) - n_val
    train_set, val_set = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False)

    model = UNet2D(in_channels=1, n_classes=NUM_CLASSES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_losses, val_dices = [], []
    epochs = 3

    for ep in range(epochs):
        model.train()
        running = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            opt.step()
            running += loss.item() * imgs.size(0)
        avg_loss = running / len(train_loader.dataset)
        train_losses.append(avg_loss)

        model.eval()
        dices = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                dices.append(dice_score(logits, masks).item())
        mean_dice = sum(dices) / len(dices)
        val_dices.append(mean_dice)
        print(f"Epoch {ep+1}/{epochs}  loss={avg_loss:.4f}  val_dice={mean_dice:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/oasis_unet.pth")

    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(val_dices, label="val dice")
    plt.legend()
    plt.savefig("training_curve.png")

if __name__ == "__main__":
    main()
