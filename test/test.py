import unittest
import model


class TestCase(unittest.TestCase):
    # def setUp(self):
    
    
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    # unittest.main()


    def suite():
        suite = unittest.TestSuite()
        suite.addTest(TestCase('test_default_widget_size'))
        suite.addTest(TestCase('test_widget_resize'))
        return suite

    runner = unittest.TextTestRunner()
    runner.run(suite())
