import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import json
import os
from PIL import Image
import logging
import matplotlib.pyplot as plt
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VehicleDamageDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        with open(annotation_file, 'r') as f:
            self.coco = json.load(f)
        self.transform = transform
        self.image_ids = [img['id'] for img in self.coco['images']]
        self.categories = {cat['id']: cat['name'] for cat in self.coco['categories']}
        self.num_classes = len(self.categories) + 1  # Add 1 for background class
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = next(img for img in self.coco['images'] if img['id'] == img_id)
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        
        anns = [ann for ann in self.coco['annotations'] if ann['image_id'] == img_id]
        
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 0 and h > 0:
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(img_info['width'], x + w)
                y2 = min(img_info['height'], y + h)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(ann['category_id'])
        
        # Ensure we have at least one box
        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]  # Add a dummy box
            labels = [0]  # Background class
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])
        
        if self.transform:
            img = self.transform(img)
        
        return img, target

def validate_data(images, targets, num_classes):
    for i, (image, target) in enumerate(zip(images, targets)):
        assert isinstance(image, torch.Tensor), f"Image {i} is not a tensor"
        assert image.dim() == 3, f"Image {i} does not have 3 dimensions"
        assert image.dtype == torch.float32, f"Image {i} is not float32"
        
        assert isinstance(target, dict), f"Target {i} is not a dictionary"
        assert "boxes" in target, f"Target {i} does not have 'boxes'"
        assert "labels" in target, f"Target {i} does not have 'labels'"
        
        boxes = target["boxes"]
        labels = target["labels"]
        
        assert isinstance(boxes, torch.Tensor), f"Boxes in target {i} is not a tensor"
        assert boxes.dim() == 2, f"Boxes in target {i} does not have 2 dimensions"
        assert boxes.shape[1] == 4, f"Boxes in target {i} does not have 4 coordinates"
        assert boxes.dtype == torch.float32, f"Boxes in target {i} is not float32"
        
        assert isinstance(labels, torch.Tensor), f"Labels in target {i} is not a tensor"
        assert labels.dim() == 1, f"Labels in target {i} is not 1-dimensional"
        assert labels.dtype == torch.int64, f"Labels in target {i} is not int64"
        
        assert boxes.shape[0] == labels.shape[0], f"Number of boxes and labels in target {i} do not match"
        
        assert (labels >= 0).all() and (labels < num_classes).all(), f"Invalid label values in target {i}"

# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create datasets
dataset_path = '/home/thomas/projects/other/vehicledamage/vehicle_damage_detection_dataset/'
train_dataset = VehicleDamageDataset(os.path.join(dataset_path, 'images/train/'), os.path.join(dataset_path, 'annotations/instances_train.json'), transform=transform)
val_dataset = VehicleDamageDataset(os.path.join(dataset_path, 'images/val/'), os.path.join(dataset_path, 'annotations/instances_val.json'), transform=transform)

# Hyperparameters
batch_size = 2
num_epochs = 10

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))

# Use CPU
device = torch.device('cpu')
logging.info(f"Using device: {device}")

# Load a pre-trained model
try:
    model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    
    num_classes = train_dataset.num_classes
    logging.info(f"Number of classes: {num_classes}")
    logging.info(f"Class mapping: {train_dataset.categories}")
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    model.to(device)
except Exception as e:
    logging.error(f"Error creating model: {str(e)}")
    raise

# Training function
def train_one_epoch(model, optimizer, data_loader, device, num_classes):
    model.train()
    total_loss = 0
    for i, (images, targets) in enumerate(data_loader):
        try:
            validate_data(images, targets, num_classes)
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            if i % 10 == 0:
                logging.info(f"Batch {i+1}/{len(data_loader)}, Loss: {losses.item():.4f}")
        except Exception as e:
            logging.error(f"Error in batch {i}: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            logging.error(f"Image shapes: {[img.shape for img in images]}")
            logging.error(f"Target shapes: {[{k: v.shape for k, v in t.items()} for t in targets]}")
            logging.error(f"Label values: {[t['labels'] for t in targets]}")
            continue

    return total_loss / len(data_loader)

# Evaluation function
def evaluate(model, data_loader, device, num_classes):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            try:
                validate_data(images, targets, num_classes)
                
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
            except Exception as e:
                logging.error(f"Error in validation batch {i}: {str(e)}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                continue

    return total_loss / len(data_loader)

# Main training loop
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

train_losses = []
val_losses = []
learning_rates = []

for epoch in range(num_epochs):
    logging.info(f"Epoch {epoch+1}/{num_epochs}")
    train_loss = train_one_epoch(model, optimizer, train_loader, device, num_classes)
    val_loss = evaluate(model, val_loader, device, num_classes)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    learning_rates.append(optimizer.param_groups[0]['lr'])
    
    logging.info(f"Epoch {epoch+1}/{num_classes} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    scheduler.step()

# Save the model
torch.save(model.state_dict(), 'vehicle_damage_detection_model.pth')

# Plot training results
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(132)
plt.plot(val_losses)
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(133)
plt.plot(learning_rates)
plt.title('Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('LR')

plt.tight_layout()
plt.savefig('training_results.png')
plt.close()

logging.info("Training completed.")