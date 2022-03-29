import torch
from typing import List
from functools import singledispatch
from cv.metrics.iou import iou
from cv.utils.nms import nms

def box_corner_to_center(boxes: torch.Tensor):
    center_x_y = (boxes[:, :2] + boxes[:, 2:]) / 2
    w_h = boxes[:, 2:] - boxes[:, :2]
    return torch.cat((center_x_y, w_h), dim = -1)

def center_to_box_corner(boxes: torch.Tensor):
    left_top = boxes[:, :2] - boxes[:,2:] / 2
    right_bottom = boxes[:, :2] + boxes[:, 2:] / 2

    return torch.cat((left_top, right_bottom), dim = 1)

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
        scores[row,:], scores[:,col] = -1, -1
    
    return anchor_labels


def offset(anchors: torch.Tensor, assigned: torch.Tensor,
            eps = 1e-6):
    anchors = box_corner_to_center(anchors)
    assigned = box_corner_to_center(assigned)

    xy_offset = 10 * (assigned[:, :2] - anchors[:, :2])/ anchors[:, 2:]
    wh_offset = 5 * torch.log(eps + assigned[:, 2:] / anchors[:, 2:])

    return torch.cat((xy_offset, wh_offset), dim = 1)

def offset_inverse(anchors: torch.Tensor, predicted_offset: torch.Tensor):
    anchors = box_corner_to_center(anchors)
    x_y = anchors[:,:2] + predicted_offset[:, :2] * anchors[:, 2:] * 0.1
    w_h = torch.exp(predicted_offset[:, 2:] * 0.2 ) * anchors[:, 2:]
    
    center = torch.cat((x_y, w_h), dim = -1)

    return center_to_box_corner(center)

def multibox_target(anchors, labels):
    """Get the offset, anchor class mask and anchor class of anchors to each batch of labels.
        Use class index 0 as background class index. Other class index add up 1.
    :param anchors: (M, 4)
    :param labels: (B, N, 5), each label (class, x1,y1,x2,y2)
    :return: (offset(B, M, 4), class_mask(B, M, 4), class(B, M))
    """
    anchors = anchors.squeeze(0)
    anchor_size = anchors.shape[0]
    classes_batch, mask_batch, offsets_batch = [],[],[]

    for i in range(labels.shape[0]):
        label = labels[i] # (N, 5)
        assigned = assign_anchor(anchors, label[:, 1:]) # (M,). Assigned label index, not class id.

        anchor_indices = assigned>=0
        label_indices = assigned[anchor_indices]
        
        # 1. classes
        classes = torch.zeros((anchor_size,), device = anchors.device, dtype = torch.long) #(M, )
        classes[anchor_indices] = label[label_indices, 0].long() + 1 # 0 is background

        # 2. mask
        mask = (assigned>=0).float().reshape(-1,1).repeat(1,4) # (M,4)

        # 3. offset
        label_boxes = torch.zeros((anchor_size,4), device = anchors.device)
        label_boxes[anchor_indices] = label[label_indices, 1:]
        offsets = offset(anchors, label_boxes) * mask #(M,4)

        classes_batch.append(classes)
        mask_batch.append(mask.reshape(-1))
        offsets_batch.append(offsets.reshape(-1))

    return [torch.stack(it) for it in (offsets_batch, mask_batch, classes_batch)]        
    
from cv.utils.nms import nms


def multibox_detection(class_predicts: torch.Tensor, predicted_offset: torch.Tensor,
                        anchors, nms_threshold=0.5, pos_threshold=.009999999):
    """Get predicted result from class distribution and offset prediction.
    :param class_predicts: (B, C, A)
    :param predicted_offset: (B, A * 4)
    :param anchors: (1, A, 4)
    :param return: (B, A, 6(class, prob/confident, box))
    """                    
    batch_size = class_predicts.shape[0]
    anchors = anchors.squeeze(0) # batch size = 1, All batches share one.

    res = []
    for i in range(batch_size):
        # Get the predicts
        class_prob = class_predicts[i][1:] # (C, A), discard background class, means every class id minus 1.
        conf, class_idx = class_prob.T.max(dim = -1) # (A, )
        predict_boxes = offset_inverse(anchors, predicted_offset[i].reshape(-1,4))
        keep_idx = nms(predict_boxes, conf, nms_threshold) # (X, 4)

        # Only preserve the keep indices, others means background.
        keep_class_idx = class_idx[keep_idx]
        class_idx[:] = -1
        class_idx[keep_idx] = keep_class_idx

        # Less than pos_threshold means background
        class_idx[conf<pos_threshold] = -1
        conf[conf<pos_threshold] = 1 - conf[conf<pos_threshold]

        #TODO: Pull keep_idx up to head

        predict = torch.cat((class_idx.reshape(-1, 1), conf.reshape(-1, 1), predict_boxes), dim = -1)
        res.append(predict)
    
    return torch.stack(res)
        

