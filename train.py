import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image

class VehicleDamageDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        with open(annotation_file, 'r') as f:
            self.coco = json.load(f)
        self.transform = transform
        self.image_ids = [img['id'] for img in self.coco['images']]
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = next(img for img in self.coco['images'] if img['id'] == img_id)
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        anns = [ann for ann in self.coco['annotations'] if ann['image_id'] == img_id]
        
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 0 and h > 0:
                # Ensure boxes are within image boundaries
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(img_info['width'], x + w)
                y2 = min(img_info['height'], y + h)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(ann['category_id'])
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])
        
        return img, target

# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create datasets
dataset_path = '/home/thomas/projects/other/vehicledamage/vehicle_damage_detection_dataset/'
train_dataset = VehicleDamageDataset(os.path.join(dataset_path, 'images/train/'), os.path.join(dataset_path, 'annotations/instances_train.json'), transform=transform)
val_dataset = VehicleDamageDataset(os.path.join(dataset_path, 'images/val/'), os.path.join(dataset_path, 'annotations/instances_val.json'), transform=transform)

# Create data loaders with batch size 1
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))

# Force CPU usage
device = torch.device('cpu')
print(f"Using device: {device}")

# Load a pre-trained model
try:
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Modify the box predictor for our number of classes
    num_classes = 8  # 7 damage classes + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    model.to(device)
except Exception as e:
    print(f"Error creating model: {str(e)}")
    raise

# Training function
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    for i, (images, targets) in enumerate(data_loader):
        try:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Skip this batch if any target has no boxes
            if any(len(t["boxes"]) == 0 for t in targets):
                print(f"Skipping batch {i} due to empty targets")
                continue

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            print(f"Batch {i+1}/{len(data_loader)}, Loss: {losses.item():.4f}")
        except Exception as e:
            print(f"Error in batch {i}: {str(e)}")
            print(f"Image shapes: {[img.shape for img in images]}")
            print(f"Target shapes: {[{k: v.shape for k, v in t.items()} for t in targets]}")
            continue

    return total_loss / len(data_loader)

# Validation function
def validate(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            try:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Skip this batch if any target has no boxes
                if any(len(t["boxes"]) == 0 for t in targets):
                    print(f"Skipping validation batch {i} due to empty targets")
                    continue

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
            except Exception as e:
                print(f"Error in validation batch {i}: {str(e)}")
                continue

    return total_loss / len(data_loader)

# Main training loop
num_epochs = 10
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss = train_one_epoch(model, optimizer, train_loader, device)
    val_loss = validate(model, val_loader, device)
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save the model
torch.save(model.state_dict(), 'vehicle_damage_detection_model.pth')

# Inference function
def inference(model, image_path, device):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(img_tensor)
    
    return prediction[0]

# Example usage of inference function
test_image_path = os.path.join(dataset_path, 'images/val/', 'some_test_image.jpg')  # Update with an actual test image path
result = inference(model, test_image_path, device)
print(result)