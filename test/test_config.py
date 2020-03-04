import unittest


class TestConfig(unittest.TestCase):

    def setUp(self):
        self.exposure_time = 0.1  # a default

    def tearDown(self):
        pass

    def test1(self):
        import scopyon.config

    def test2(self):
        from scopyon.config import DefaultConfiguration
        config = DefaultConfiguration()

    def test3(self):
        from scopyon.config import DefaultConfiguration
        config = DefaultConfiguration()
        self.assertAlmostEqual(config.default.detector.exposure_time, self.exposure_time)
        config.default.detector.exposure_time = 0.033
        self.assertAlmostEqual(config.default.detector.exposure_time, 0.033)

    def test4(self):
        from scopyon.config import DefaultConfiguration
        config = DefaultConfiguration()
        config.update("""
        default:
            detector:
                exposure_time: 0.033
        """)
        self.assertAlmostEqual(config.default.detector.exposure_time, 0.033)

        config.update("""
        default:
            detector:
                exposure_time:
                    value: 0.1
        """)
        self.assertAlmostEqual(config.default.detector.exposure_time, 0.1)

    def test4(self):
        from scopyon.config import DefaultConfiguration
        config = DefaultConfiguration()
        config.update("""
        default:
            detector:
                exposure_time:
                    value: 0.033
                    units: s
        """)
        self.assertAlmostEqual(config.default.detector.exposure_time, 0.033)

        config.update("""
        default:
            detector:
                exposure_time:
                    value: 100
                    units: ms
        """)
        self.assertAlmostEqual(config.default.detector.exposure_time, 0.1)

        config.default.detector.exposure_time = 0.033
        self.assertAlmostEqual(config.default.detector.exposure_time, 0.033)

    def test5(self):
        from scopyon.config import DefaultConfiguration
        from scopyon.constants import Q_
        config = DefaultConfiguration()
        config.default.detector.exposure_time = Q_(33, 'ms')
        self.assertAlmostEqual(config.default.detector.exposure_time, 0.033)

    def test5(self):
        from scopyon.config import DefaultConfiguration
        from scopyon.constants import Q_
        config = DefaultConfiguration()

        from pint.errors import DimensionalityError
        with self.assertRaises(DimensionalityError):
            config.default.detector.exposure_time = Q_(33, 'm')


if __name__ == '__main__':
    unittest.main()
