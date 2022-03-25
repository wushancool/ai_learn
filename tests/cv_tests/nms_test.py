import unittest
import numpy as np
from cv.utils.nms import nms

class NmsTests(unittest.TestCase):

    def test_nms(self):
        boxes = np.array([(187, 82, 337, 317), (150, 67, 305, 282), (246, 121, 368, 304)])
        score = np.array([0.9, 0.65, 0.8])
        threshold = 0.3
        
        picked = nms(boxes, score, threshold)

        self.assertEqual(picked, [0])

