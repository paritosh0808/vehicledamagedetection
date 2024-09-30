import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from dataset_class import VehicleDamageDataset
from training_pipeline import get_model
from torchvision import transforms


def compute_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection / (area1 + area2 - intersection + 1e-6)
    return iou


def evaluate_model(model, data_loader, device, iou_threshold=0.5):
    model.eval()

    all_predictions = []
    all_targets = []

    # Define transform to convert PIL Image to tensor
    to_tensor = transforms.ToTensor()

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            # Convert PIL Images to tensors and move to device
            images = [to_tensor(img).to(device) for img in images]

            outputs = model(images)

            for output, target in zip(outputs, targets):
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()

                # Filter predictions based on a score threshold (e.g., 0.5)
                mask = scores > 0.5
                boxes = boxes[mask]
                scores = scores[mask]
                labels = labels[mask]

                all_predictions.append({
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels
                })

                all_targets.append({
                    'boxes': target['boxes'].cpu().numpy(),
                    'labels': target['labels'].cpu().numpy()
                })

    # Compute mAP
    ap_per_class = defaultdict(list)
    for pred, target in zip(all_predictions, all_targets):
        for class_id in np.unique(np.concatenate([pred['labels'], target['labels']])):
            class_pred_indices = pred['labels'] == class_id
            class_target_indices = target['labels'] == class_id

            class_pred_boxes = pred['boxes'][class_pred_indices]
            class_pred_scores = pred['scores'][class_pred_indices]
            class_target_boxes = target['boxes'][class_target_indices]

            if len(class_pred_boxes) == 0 and len(class_target_boxes) == 0:
                ap_per_class[class_id].append(1.0)  # Perfect score if no predictions and no targets
                continue
            if len(class_pred_boxes) == 0 or len(class_target_boxes) == 0:
                ap_per_class[class_id].append(0.0)  # Zero score if predictions or targets are missing
                continue

            # Compute IoU for all combinations of predicted and target boxes
            ious = np.zeros((len(class_pred_boxes), len(class_target_boxes)))
            for i, pred_box in enumerate(class_pred_boxes):
                for j, target_box in enumerate(class_target_boxes):
                    ious[i, j] = compute_iou(pred_box, target_box)

            # Compute precision and recall
            sorted_indices = np.argsort(-class_pred_scores)
            tp = np.zeros(len(class_pred_boxes))
            fp = np.zeros(len(class_pred_boxes))

            matched_targets = set()
            for i in sorted_indices:
                best_iou = 0
                best_j = -1
                for j in range(len(class_target_boxes)):
                    if j in matched_targets:
                        continue
                    if ious[i, j] > best_iou:
                        best_iou = ious[i, j]
                        best_j = j

                if best_iou >= iou_threshold:
                    tp[i] = 1
                    matched_targets.add(best_j)
                else:
                    fp[i] = 1

            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            recalls = tp_cumsum / len(class_target_boxes)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

            # Compute AP using 11-point interpolation
            ap = 0
            for t in np.arange(0, 1.1, 0.1):
                if np.sum(recalls >= t) == 0:
                    p = 0
                else:
                    p = np.max(precisions[recalls >= t])
                ap += p / 11

            ap_per_class[class_id].append(ap)

    # Compute mAP
    mAP = np.mean([np.mean(aps) for aps in ap_per_class.values()])

    # Compute per-class AP
    class_ap = {class_id: np.mean(aps) for class_id, aps in ap_per_class.items()}

    return mAP, class_ap


def main():
    test_img_dir = '/home/thomas/projects/other/vehicledamage/vehicle_damage_detection_dataset/images/val'
    test_ann_file = '/home/thomas/projects/other/vehicledamage/vehicle_damage_detection_dataset/annotations/instances_val.json'

    num_classes = 8

    test_set = VehicleDamageDataset(test_img_dir, test_ann_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes)
    model.to(device)

    # Load the trained model
    checkpoint = torch.load('best_vehicle_damage_model.pth')
    model.load_state_dict(checkpoint)
    model.to(device)

    # Create a DataLoader for the test set
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

    mAP, class_ap = evaluate_model(model, test_loader, device)

    print(f"Overall mAP: {mAP:.4f}")
    print("Per-class AP:")
    for class_id, ap in class_ap.items():
        print(f"Class {class_id}: {ap:.4f}")


if __name__ == '__main__':
    main()
