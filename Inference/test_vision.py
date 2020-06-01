import unittest
import vision

# tests should be isolated


class TestVision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("setUpClass")

    @classmethod
    def tearDownClass(cls):
        print("tearDownClass")

    def setUp(self):
        # runs before every single test
        pass

    def tearDown(self):
        # after every test
        pass

    def test_load_model(self):
        model = vision.load_model('alexnet')
        self.assertIsNotNone(model)
        model = vision.load_model('alexnet1')
        self.assertIsNone(model)

    def test_dummy(self):
        self.assertEqual(5, 5)


if __name__ == '__main__':
    unittest.main()
