import unittest
from cv.metrics.iou import iou

class IouTests(unittest.TestCase):

    def test_iou_normal(self):
        true_bbox, predict_bbox = [100, 35, 398, 400], [40, 150, 355, 398]
        res = iou(true_bbox, predict_bbox)
        # Order not matter.
        res1 = iou(predict_bbox, true_bbox)
        self.assertEqual(res, res1)
        self.assertAlmostEqual(res, 0.511443, places = 5)

    
    def test_iou_no_insection(self):
        true_bbox, predict_bbox = [1,1,2,2], [3,3,4,4]
        res = iou(true_bbox, predict_bbox)
        self.assertEqual(res, 0)
    
    def test_iou_insection_corner(self):
        b1, b2 = [1,1,2,2], [2,2,3,3]
        res = iou(b1, b2)
        self.assertEqual(res, 0)
