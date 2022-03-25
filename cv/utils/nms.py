import numpy as np
from cv.metrics.iou import iou
from functools import partial

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
        qualified = np.apply_along_axis(iou_func,1, boxes[candidates]) < threshold
        candidates = candidates[qualified]

    return selected