import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class VehicleDamageDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        
        self.images = annotations['images']
        self.annotations = annotations['annotations']
        self.categories = {cat['id']: cat['name'] for cat in annotations['categories']}
        
        # Filter out images without valid annotations
        self.valid_image_ids = set()
        for ann in self.annotations:
            if ann['category_id'] > 0:  # Exclude 'severity-damage'
                self.valid_image_ids.add(ann['image_id'])
            else:
                pass
        self.images = [img for img in self.images if img['id'] in self.valid_image_ids]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get annotations for this image
        img_annotations = [ann for ann in self.annotations if ann['image_id'] == img_info['id'] and ann['category_id'] > 0]
        
        boxes = []
        labels = []
        for ann in img_annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            labels.append(ann['category_id'] - 1)  # Adjust category_id
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        
        return image, target