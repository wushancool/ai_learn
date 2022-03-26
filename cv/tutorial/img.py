import matplotlib.pyplot as plt
import math
from voc import NamedBoundedBox
from matplotlib import patches
from PIL import Image
from typing import List

def imgs_show(imgs, rows):
    cols = math.floor(len(imgs)/rows)
    plt.figure(figsize = (10, 2* rows))
    for i, im in enumerate(imgs):
        ax = plt.subplot(rows, cols, i+1)
        ax.axis("off")
        ax.imshow(im, interpolation="nearest")
    plt.tight_layout(pad = 0.2)

def img_withbox(img_file: str, boxes: List[NamedBoundedBox]):
    fig, ax = plt.subplots(1)
    image =Image.open(img_file)
    ax.imshow(image)
    colors = plt.cm.get_cmap("hsv", 10)

    def add_patch(box,name, color):
        rectangle = patches.Rectangle((box.xmin, box.ymin), width=box.width,
                                       height=box.height, 
                                       edgecolor=color,label="1",
                                       fill = False,linewidth=1)
        ax.text(box.xmin,box.ymin-3, s=name, color=color)
        ax.add_patch(rectangle)
        
    [add_patch(it.box, it.name, colors(i)) 
                    for i, it in enumerate(boxes)]
    
    return image