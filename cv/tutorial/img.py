import matplotlib.pyplot as plt
import math

def imgs_show(imgs, rows):
    cols = math.floor(len(imgs)/rows)
    plt.figure(figsize = (10, 2* rows))
    for i, im in enumerate(imgs):
        ax = plt.subplot(rows, cols, i+1)
        ax.axis("off")
        ax.imshow(im, interpolation="nearest")
    plt.tight_layout(pad = 0.2)
