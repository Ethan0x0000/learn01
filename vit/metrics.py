import torch


def _to_device_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def _get_labels_from_logits(logits):
    if logits.ndim == 1:
        return logits
    return logits.argmax(dim=1)


def accuracy_from_logits(logits, targets):
    logits = _to_device_tensor(logits)
    targets = _to_device_tensor(targets)
    preds = _get_labels_from_logits(logits)
    preds = preds.view(-1)
    targets = targets.view(-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    if total == 0:
        return 0.0
    return correct / total


def confusion_matrix(preds, targets, num_classes):
    preds = _to_device_tensor(preds).view(-1)
    targets = _to_device_tensor(targets).view(-1)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets, preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t.long(), p.long()] += 1
    return cm


def confusion_matrix_from_logits(logits, targets, num_classes):
    logits = _to_device_tensor(logits)
    preds = _get_labels_from_logits(logits)
    return confusion_matrix(preds, targets, num_classes)


def precision_recall_f1_from_cm(cm, average="macro"):
    cm = _to_device_tensor(cm)
    num_classes = cm.size(0)
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    precision = tp.float() / (tp + fp).clamp(min=1).float()
    recall = tp.float() / (tp + fn).clamp(min=1).float()
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    if average == "macro":
        return {
            "precision": precision.mean().item(),
            "recall": recall.mean().item(),
            "f1": f1.mean().item(),
        }
    if average == "micro":
        tp_sum = tp.sum().float()
        fp_sum = fp.sum().float()
        fn_sum = fn.sum().float()
        precision_micro = tp_sum / (tp_sum + fp_sum).clamp(min=1)
        recall_micro = tp_sum / (tp_sum + fn_sum).clamp(min=1)
        f1_micro = (
            2
            * precision_micro
            * recall_micro
            / (precision_micro + recall_micro).clamp(min=1e-8)
        )
        return {
            "precision": precision_micro.item(),
            "recall": recall_micro.item(),
            "f1": f1_micro.item(),
        }
    raise ValueError("average must be 'macro' or 'micro'")


def precision_recall_f1_from_logits(logits, targets, num_classes, average="macro"):
    cm = confusion_matrix_from_logits(logits, targets, num_classes)
    return precision_recall_f1_from_cm(cm, average=average)


def classification_metrics_from_logits(logits, targets, num_classes, average="macro"):
    logits = _to_device_tensor(logits)
    targets = _to_device_tensor(targets)
    acc = accuracy_from_logits(logits, targets)
    cm = confusion_matrix_from_logits(logits, targets, num_classes)
    prf = precision_recall_f1_from_cm(cm, average=average)
    return {
        "accuracy": acc,
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(cm, class_names=None, normalize=False, cmap="Blues", save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    cm_np = _to_device_tensor(cm).cpu().numpy()
    if normalize:
        row_sums = cm_np.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_np = cm_np.astype(float) / row_sums
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm_np, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax)
    if class_names is not None:
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm_np.max() / 2.0 if cm_np.size > 0 else 0.0
    for i in range(cm_np.shape[0]):
        for j in range(cm_np.shape[1]):
            value = cm_np[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}" if normalize else str(int(value)),
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
            )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


def _to_tensor(x, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype)


def box_iou(boxes1, boxes2):
    boxes1 = _to_tensor(boxes1)
    boxes2 = _to_tensor(boxes2)
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(0, 0)
    b1 = boxes1.unsqueeze(1)
    b2 = boxes2.unsqueeze(0)
    inter_x1 = torch.max(b1[..., 0], b2[..., 0])
    inter_y1 = torch.max(b1[..., 1], b2[..., 1])
    inter_x2 = torch.min(b1[..., 2], b2[..., 2])
    inter_y2 = torch.min(b1[..., 3], b2[..., 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(
        min=0
    )
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(
        min=0
    )
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
    union_area = union_area.clamp(min=1e-8)
    return inter_area / union_area


def box_iou_batch(pred_boxes, true_boxes):
    return box_iou(pred_boxes, true_boxes)


def compute_ap(recall, precision):
    recall = _to_tensor(recall)
    precision = _to_tensor(precision)
    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
    for i in range(mpre.size(0) - 1, 0, -1):
        mpre[i - 1] = torch.max(mpre[i - 1], mpre[i])
    idx = torch.where(mrec[1:] != mrec[:-1])[0]
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap.item()


def compute_tp_fp_for_class(pred_boxes, true_boxes, iou_threshold=0.5):
    if len(pred_boxes) == 0:
        return [], 0
    pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x[2], reverse=True)
    image_to_gts = {}
    for img_id, _, x1, y1, x2, y2 in true_boxes:
        image_to_gts.setdefault(img_id, [])
        image_to_gts[img_id].append([x1, y1, x2, y2, 0])
    tps = []
    fps = []
    total_gts = len(true_boxes)
    for img_id, _, score, x1, y1, x2, y2 in pred_boxes_sorted:
        gts = image_to_gts.get(img_id, [])
        if len(gts) == 0:
            tps.append(0)
            fps.append(1)
            continue
        pred_box = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        gt_boxes = torch.tensor([[g[0], g[1], g[2], g[3]] for g in gts], dtype=torch.float32)
        ious = box_iou(pred_box, gt_boxes).squeeze(0)
        best_iou, best_idx = ious.max(0)
        if best_iou.item() >= iou_threshold and gts[best_idx][4] == 0:
            tps.append(1)
            fps.append(0)
            gts[best_idx][4] = 1
        else:
            tps.append(0)
            fps.append(1)
    return list(zip(tps, fps)), total_gts


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5):
    classes = set()
    for _, c, _, _, _, _, _ in pred_boxes:
        classes.add(c)
    for _, c, _, _, _, _ in true_boxes:
        classes.add(c)
    aps = []
    for cls in classes:
        cls_preds = [b for b in pred_boxes if b[1] == cls]
        cls_trues = [b for b in true_boxes if b[1] == cls]
        pairs, total_gts = compute_tp_fp_for_class(
            cls_preds, cls_trues, iou_threshold=iou_threshold
        )
        if total_gts == 0:
            continue
        tps = torch.tensor([p[0] for p in pairs], dtype=torch.float32)
        fps = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
        tps_cum = torch.cumsum(tps, dim=0)
        fps_cum = torch.cumsum(fps, dim=0)
        recalls = tps_cum / total_gts
        precisions = tps_cum / (tps_cum + fps_cum).clamp(min=1e-8)
        ap = compute_ap(recalls, precisions)
        aps.append(ap)
    if len(aps) == 0:
        return 0.0
    return sum(aps) / len(aps)

