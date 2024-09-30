import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import random
from dataset_class import VehicleDamageDataset
from utils import visualize_sample, validate

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch, writer):
    model.train()
    total_loss = 0
    
    for i, (images, targets) in enumerate(tqdm(data_loader, desc=f"Training Epoch {epoch}")):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Skip this batch if any target has no boxes
        if any(len(target['boxes']) == 0 for target in targets):
            continue
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train_step', losses.item(), epoch * len(data_loader) + i)
    
    avg_loss = total_loss / len(data_loader)
    writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
    return avg_loss


def main():
    # Configuration
    train_img_dir = os.getenv('TRAIN_IMG_DIR')
    train_ann_file = os.getenv('TRAIN_ANN_FILE')
    val_img_dir = os.getenv('VAL_IMG_DIR')
    val_ann_file = os.getenv('VAL_ANN_FILE')

    num_classes = 8  
    num_epochs = int(os.getenv('NUM_EPOCHS', '50'))
    batch_size = int(os.getenv('BATCH_SIZE', '4'))
    learning_rate = float(os.getenv('LEARNING_RATE', '0.005'))
    num_train_images = int(os.getenv('NUM_TRAIN_IMAGES', '1000'))

    writer = SummaryWriter(os.getenv('TENSORBOARD_LOG_DIR', 'runs/vehicle_damage_detection'))

    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    full_train_dataset = VehicleDamageDataset(train_img_dir, train_ann_file, transform=transform)
    train_indices = random.sample(range(len(full_train_dataset)), num_train_images)
    
    train_set = Subset(full_train_dataset, train_indices)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

    val_set = VehicleDamageDataset(val_img_dir, val_ann_file, transform=transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))


    # Model
    model = get_model(num_classes)
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)


    checkpoint_path = os.getenv('CHECKPOINT_PATH', 'best_vehicle_damage_model.pth')
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
        else:
            # Old format (directly saved state_dict)
            model.load_state_dict(checkpoint)
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    best_val_accuracy = 0
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, writer)
        val_accuracy = validate(model, val_loader, device, epoch, writer)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_vehicle_damage_model.pth')
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.4f}")

        
        if epoch % 5 == 0:
            model.eval()
            sample_images, sample_targets = next(iter(val_loader))
            sample_image = sample_images[0].to(device)
            sample_target = sample_targets[0]  
            with torch.no_grad():
                prediction = model([sample_image])[0]
            visualize_sample(sample_image, sample_target, prediction, full_train_dataset.categories, writer, epoch)

    
    # Save the model
    torch.save(model.state_dict(), os.getenv('FINAL_MODEL_PATH', 'vehicle_damage_model.pth'))

if __name__ == '__main__':
    main()





















