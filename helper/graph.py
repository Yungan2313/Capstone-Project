import matplotlib.pyplot as plt

def draw_loss(train_losses, val_losses, run_dir):
    # 畫出 Loss 圖
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(run_dir / "loss_curve.png")  # 儲存圖片
