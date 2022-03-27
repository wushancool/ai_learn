import torch
from typing import List
from functools import singledispatch
from cv.metrics.iou import iou

@singledispatch
def multibox_prior(input_shape: List[int], 
                    sizes: List[float], 
                    ratios: List[float])-> torch.Tensor:
    """
    Generate all anchor boxes with an image based on sizes and ratios. 
    :param input_shape: (height, width). The shape of input image.
    :param sizes: (int, ). 
    :param ratios: (int, ). 
    :return: ((sizes + ratios - 1) * height * width, 4). The coordinates of all anchor boxes.
    """
    sizes, ratios = torch.tensor(sizes), torch.tensor(ratios)
    height, width = input_shape

    # 1. The coordinates of center in all points.
    center_offset = 0.5
    h_offset = (torch.arange(height) + center_offset) / height
    w_offset = (torch.arange(width) + center_offset) / width
    y_shift,x_shift  = torch.meshgrid(h_offset, w_offset) # direction matters
    # (w,h)
    x_shift, y_shift = x_shift.reshape(-1), y_shift.reshape(-1)

    # 2. All anchor boxes of single pixel, using height and width
    w = torch.cat((sizes * torch.sqrt(ratios[0]),sizes[0] * torch.sqrt(ratios[1:])))\
             * height / width # When r = 1, anchor box is square
    h = torch.cat((sizes / torch.sqrt(ratios[0]), sizes[0] / torch.sqrt(ratios[1:])))
    w_h_pixel = torch.stack((-w, -h, w, h)).T / 2 # (N, 4), 4 is the offset
    
    # 3. center + h/w => left-top/right-bottom coordinates
    # Every pixel combine h/w and center shift to coordinate.
    w_h_pixels = w_h_pixel.repeat(height * width, 1) # (all, 4)
    shifts = torch.stack((x_shift, y_shift, x_shift, y_shift),1).repeat_interleave(len(w), dim = 0) #(all,4)

    return w_h_pixels + shifts
    


@multibox_prior.register
def _(input: torch.Tensor,
                    sizes: List[float], 
                    ratios: List[float])->torch.Tensor:
    res = multibox_prior(input.shape[-2:], sizes, ratios)
    return res.to(input.device).unsqueeze(0)

def assign_anchor(anchors, ground_truth_boxes, iou_threshold=0.5):
    """ Assign a class to each anchor by criteria.
    :param anchors: (M, 4).Candidates
    :param ground_truth_boxes:(N,4). labels
    :param iou_threshold:
    :return: (M,)
    """
    scores = iou(anchors, ground_truth_boxes)
    anchor_labels = torch.full((len(anchors),), -1, device = anchors.device)
    # 1. Every anchor will have its max iou score label.
    max_scores, indices = torch.max(scores, dim = 1)
    qualified = torch.nonzero(max_scores>= iou_threshold).squeeze(-1)
    anchor_labels[qualified] = indices[qualified]

    # 2. Give every ground-truth one most likely anchor.
    label_count = len(ground_truth_boxes)
    for _ in range(label_count):
        # Find max, (consider threshold? Rare to see in practice?)
        max_index = scores.argmax()
        col = max_index % label_count
        row = torch.div(max_index , label_count, rounding_mode = "floor")
        anchor_labels[row] = col

        # discard row and col
        scores[row], scores[col] = -1, -1
    
    return anchor_labels