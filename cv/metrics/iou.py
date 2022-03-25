import numpy as np

def iou(box1, box2, wh=False):
    """
    Intersection over Union (IoU) for object detection.
    IoU = Area of Intersection of two boxes / Area of Union of two boxes.
    Area of Union = area_box1 + area_box2 - area_intersection

    @param box1: If wh is `True`, box1.shape is (x_c, y_c, width, height),
                 else box1.shape is (x_min, y_min, x_max, y_max)
    @param box2: Same as box1, another box.
    @param If wh is `True`, 
    """

    def transform(b, wh):
        if wh:
            x_c, y_c, width, height = b
            return x_c - width/2, y_c - height/2, x_c + width/2, y_c + height/2
        else:
            return b

    b1 = transform(box1, wh)
    b2 = transform(box2, wh)
    b_in = max(b1[0], b2[0]), max(b1[1], b2[1]),\
           min(b1[2], b2[2]), min(b1[3], b2[3])
    
    area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    area_in = area(b_in)
    area_union = area(b1) + area(b2) - area_in

    return area_in / area_union
    