import unittest
import vision


class TestVision(unittest.TestCase):

    def test_load_model(self):
        model = vision.load_model('alexnet')
        self.assertIsNotNone(model)

    def test_load_model_fail(self):
        model = vision.load_model('alexnet1')
        self.assertIsNone(model)
