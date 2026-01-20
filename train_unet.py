from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 50

class KSDD2SegDataset(Dataset) : 
    def __init__(self, list_path, transform = None) : 
        # list_path: train_list.txt or val_list.txt construct using split_dataset.py
        with open(list_path, "r") as file : 
            # read absolute path of all images
            self.image_paths = [line.strip() for line in file.readlines() if line.strip()]
        
        self.transform = transform

    def __len__(self) : 
        return len(self.image_paths)

    def __getitem__(self, idx) : 
        # 1. get image path
        img_path = self.image_paths[idx]

        # 2. build ground truth path
        mask_path = img_path.replace("/images/", "/ground_truth/").replace(".png", "_GT.png")

        # 3. read image and ground truth
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask > 127, 1, 0).astype(np.float32) # convert to 0: background, 1: defect

        # 4. data augmentation
        if self.transform : 
            augmented = self.transform(image = image, mask = mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else : 
            # basic preprocessing (transform to tensor and (C, H, W))
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float() # (1, H, W)
        
        return image, mask

def calculate_dice(pr, gt, threshold=0.5) : 
    """Dice Coefficient: 2*|A∩B| / (|A|+|B|)"""
    # transform predict result to 0 ~ 1
    pr = torch.sigmoid(pr)
    # if result > 0.5 then defect
    pr = (pr > threshold).float()

    # flatten
    pr_flat = pr.view(pr.size(0), -1)
    gt_flat = gt.view(gt.size(0), -1)
    
    # intersection
    intersection = (pr_flat * gt_flat).sum(dim = 1)
    # sum value
    sum_val = pr_flat.sum(dim = 1) + gt_flat.sum(dim = 1)
    
    # dice = 2 * intersection / total
    dice = (2.0 * intersection + 1e-7) / (sum_val + 1e-7)

    # return batch average
    return dice.mean().item()

if __name__ == "__main__" : 
    # train dataset transform
    train_transform = A.Compose([
        # letterbox logic
        A.LongestMaxSize(max_size = 640),  # resize long side to 640
        A.PadIfNeeded(                     # padding with black pixel
            min_height = 640, 
            min_width = 640, 
            border_mode = 0, 
            fill = (0, 0, 0), 
            fill_mask = 0
        ), 
        
        # data augmentation
        A.HorizontalFlip(p = 0.5),         # flip left/right (50%)
        A.VerticalFlip(p = 0.5),           # flip up/down (50%)
        A.RandomRotate90(p = 0.5),         # random rotate 90 degree (50%)
        A.ShiftScaleRotate(                # random shift / scale / rotate (50%)
            shift_limit = 0.0625, 
            scale_limit = 0.1, 
            rotate_limit = 15, 
            p = 0.5
        ), 
        
        # normalize
        A.Normalize(mean = (0, 0, 0), std = (1, 1, 1)), 
        ToTensorV2(), 
    ])

    # val dataset transform
    val_transform = A.Compose([
        # letterbox logic
        A.LongestMaxSize(max_size = 640),  # resize long side to 640
        A.PadIfNeeded(                     # padding with black pixel
            min_height = 640, 
            min_width = 640, 
            border_mode = 0, 
            fill = (0, 0, 0), 
            fill_mask = 0
        ), 
        
        # normalize
        A.Normalize(mean = (0, 0, 0), std = (1, 1, 1)), 
        ToTensorV2(), 
    ])

    # build DataLoader
    train_dataset = KSDD2SegDataset("/home/neat/deep_learning_hw2/KSDD2/train_list.txt", transform = train_transform)
    val_dataset = KSDD2SegDataset("/home/neat/deep_learning_hw2/KSDD2/val_list.txt", transform = val_transform)
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4, pin_memory = True)
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4, pin_memory = True)

    # model definition
    model = smp.Unet(
        encoder_name = "resnet50",     # use ResNet50 as backbone
        encoder_weights = "imagenet",  # load pre-trained weight
        in_channels = 3,               # RGB input
        classes = 1,                   # one class (defect or not)
        activation = None              # use BCEWithLogitsLoss
    ).to(DEVICE)

    # loss function
    criterion = smp.losses.DiceLoss(mode = "binary", from_logits = True)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)
    scaler = torch.amp.GradScaler("cuda")

    # loop training
    best_dice = 0.0
    history = {"train_loss": [], "val_dice": []}

    for epoch in range(EPOCHS) : 
        # train
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc = f"Epoch {epoch+1}/{EPOCHS} [Train]", bar_format = '{l_bar}{bar:30} {r_bar}', ascii = ' ▮')

        for imgs, masks in train_pbar : 
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda") : 
                outputs = model(imgs)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            # train_pbar.set_postfix(loss = loss.item())

        # val
        model.eval()
        total_val_dice = 0
        val_pbar = tqdm(val_loader, desc = f"Epoch {epoch+1}/{EPOCHS} [Val]  ", bar_format = '{l_bar}{bar:30} {r_bar}', ascii = ' ▮')

        with torch.no_grad() : 
            for imgs, masks in val_pbar : 
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                outputs = model(imgs)

                dice_score = calculate_dice(outputs, masks)
                total_val_dice += dice_score
                # val_pbar.set_postfix(dice = dice_score)

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_dice = total_val_dice / len(val_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_dice"].append(avg_val_dice)

        print(f"Summary Epoch {epoch+1}: Train Loss={avg_train_loss:.3f}, Val Dice={avg_val_dice:.3f}\n")

        # save best model
        if avg_val_dice > best_dice : 
            best_dice = avg_val_dice
            torch.save(model.state_dict(), "best_unet_ksdd2.pth")
        
    # plot dice trend
    plt.figure(figsize = (10, 5))
    plt.plot(range(1, EPOCHS+1), history["val_dice"], marker = 'o', color = 'teal', label = 'Validation Dice')
    plt.axhline(y = max(history["val_dice"]), color = 'r', linestyle = '--', label = f'Best Dice: {max(history["val_dice"]):.3f}')
    plt.title('Performance Evaluation: Dice Coefficient per Epoch', fontsize = 14)
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.savefig("performance_dice_trend.png")
    print("performance_dice_trend.png saved.")

    # plot dice distribution
    all_dice_scores = []
    with torch.no_grad() : 
        for imgs, masks in val_loader : 
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs)
            for j in range(imgs.size(0)):
                dice_score = calculate_dice(outputs[j:j+1], masks[j:j+1])
                all_dice_scores.append(dice_score)
    plt.figure(figsize = (8, 6))
    plt.hist(all_dice_scores, bins = 20, color = 'skyblue', edgecolor = 'black')
    plt.title('Distribution of Dice Scores across Validation Set', fontsize = 14)
    plt.xlabel('Dice Score')
    plt.ylabel('Number of Images')
    plt.axvline(np.mean(all_dice_scores), color = 'red', linestyle = 'dashed', label = f'Mean Dice: {np.mean(all_dice_scores):.3f}')
    plt.legend()
    plt.savefig("dice_distribution.png")
    print("dice_distribution.png saved.")