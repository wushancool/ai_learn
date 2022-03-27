import numpy as np
from cv.metrics.iou import iou
from functools import partial, singledispatch
import torch

@singledispatch
def nms(boxes: np.ndarray, score: np.ndarray, threshold:float):
    """
    NMS of single class.

    :param bboxes: (batch, 4), all selected boxes locations.
    :param confidence_score: (batch, ), score of every box
    :param threshold: The threshold of IoU score.
    :return: list, The indices which picked
    """
    
    candidates = np.argsort(score)

    selected = []
    while len(candidates) > 0:
        # Select top1 score
        top1 = candidates[-1]
        selected.append(top1)

        # Others will be removed where IoU score less than `threshold` with top1.
        # The `top1` will be removed because the IoU will be 1.
        # The iou function is a single item compare.
        iou_func = partial(iou, box2 = boxes[top1])
        qualified = np.apply_along_axis(iou_func,1, boxes[candidates]) <= threshold
        candidates = candidates[qualified]

    return selected

@nms.register
def _(boxes: torch.Tensor, score: torch.Tensor, threshold: float):
    rank = score.argsort(dim = -1, descending = True)
    
    keep = []
    while len(rank) > 0:
        top1 = rank[0]
        keep.append(top1)

        iou_scores = iou(boxes[top1].unsqueeze(0), boxes[rank[1:]].reshape(-1,4)).reshape(-1)
        qualified = torch.nonzero(iou_scores <= threshold).reshape(-1) 
        rank = rank[qualified + 1]

    return torch.tensor(keep, device = boxes.device)