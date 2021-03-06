import matplotlib.pyplot as plt
import math
from voc import NamedBoundedBox, BoundedBox
from matplotlib import patches
from PIL import Image
from typing import List, Optional, Union
from functools import singledispatch

def imgs_show(imgs, rows):
    cols = math.floor(len(imgs)/rows)
    axes = []
    plt.figure(figsize = (10, 2* rows))
    for i, im in enumerate(imgs):
        ax = plt.subplot(rows, cols, i+1)
        ax.axis("off")
        ax.imshow(im, interpolation="nearest")
        axes.append(ax)
    plt.tight_layout(pad = 0.2)

    return axes

def add_patches(ax, boxes, names):
    colors = plt.cm.get_cmap("Set1", 10)
    if boxes is None:
        boxes = []
    if len(boxes)>0 and not isinstance(boxes[0], NamedBoundedBox):
        if names is None:
            names = [""] * len(boxes)
        boxes = [NamedBoundedBox(name , BoundedBox(*b)) for name, b in zip(names,boxes)]

    def add_patch(box,name, color):
        rectangle = patches.Rectangle((box.xmin, box.ymin), width=box.width,
                                       height=box.height, 
                                       edgecolor=color,
                                       fill = False,linewidth=1.5)
        ax.text(box.xmin, box.ymin-8, s=name, color="white", backgroundcolor=color, fontsize = 9)
        ax.add_patch(rectangle)
        
    [add_patch(it.box, it.name, colors(i)) 
                    for i, it in enumerate(boxes)]

@singledispatch
def img_withbox(img_file: str, 
                boxes: Union[List[NamedBoundedBox] , List[List[int]]]= None,
                names = None):
    fig, ax = plt.subplots(1)
    image =Image.open(img_file)

    ax.imshow(image)
    add_patches(ax, boxes, names)
    return image

