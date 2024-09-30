import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_accuracy(predictions, targets, iou_threshold=0.5, score_threshold=0.5):
    total_correct = 0
    total_objects = 0
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']
        
        target_boxes = target['boxes']
        target_labels = target['labels']
        
        # Filter predictions based on score threshold
        mask = pred_scores > score_threshold
        pred_boxes = pred_boxes[mask]
        pred_labels = pred_labels[mask]
        
        total_objects += len(target_boxes)
        
        if len(pred_boxes) == 0:
            continue
        
        # Compute IoU between predicted and target boxes
        ious = box_iou(pred_boxes, target_boxes)
        
        # Find the best prediction for each target
        max_ious, max_idx = ious.max(dim=0)
        
        # Count correct predictions (IoU > threshold and correct class)
        correct = (max_ious > iou_threshold) & (pred_labels[max_idx] == target_labels)
        total_correct += correct.sum().item()
    
    return total_correct / total_objects if total_objects > 0 else 0

def validate(model, data_loader, device, epoch, writer):
    model.eval()
    total_accuracy = 0
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(data_loader, desc=f"Validating Epoch {epoch}")):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Skip this batch if any target has no boxes
            if any(len(target['boxes']) == 0 for target in targets):
                continue
            
            outputs = model(images)
            
            accuracy = compute_accuracy(outputs, targets)
            total_accuracy += accuracy
            
            # Log to TensorBoard
            writer.add_scalar('Accuracy/val_step', accuracy, epoch * len(data_loader) + i)
    
    avg_accuracy = total_accuracy / len(data_loader)
    writer.add_scalar('Accuracy/val_epoch', avg_accuracy, epoch)
    return avg_accuracy

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou

def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])



def visualize_sample(image, target, prediction, categories, writer, step):
    image = image.cpu()
    boxes = target['boxes'].cpu()
    labels = target['labels'].cpu()
    target = {k: v.cpu() for k, v in target.items()}
    prediction = {k: v.cpu() for k, v in prediction.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Ground truth
    ax1.imshow(F.to_pil_image(image))
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='green')
        ax1.add_patch(rect)
        ax1.text(x1, y1, categories[label.item()+1], color='green')
    ax1.set_title('Ground Truth')
    
    # Prediction
    ax2.imshow(F.to_pil_image(image))
    for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
        if score > 0.5:  # Only show predictions with confidence > 0.5
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red')
            ax2.add_patch(rect)
            ax2.text(x1, y1, f"{categories[label.item()+1]}: {score:.2f}", color='red')
    ax2.set_title('Prediction')
    
    writer.add_figure('Sample Prediction', fig, step)
    plt.close(fig)