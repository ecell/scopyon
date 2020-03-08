import unittest


class TestEPIFM(unittest.TestCase):

    def setUp(self):
        self.radial_cutoff = 1000.0e-9
        self.radial_resolution = 1.0e-9

    def tearDown(self):
        pass

    def test1(self):
        import scopyon._epifm

    def test2(self):
        print('Testing TRITC ...')
        import numpy
        from scopyon._epifm import PointSpreadingFunction
        psf = PointSpreadingFunction(psf_radial_cutoff=self.radial_cutoff, psf_radial_width=None, psf_depth_cutoff=1000.0e-9, fluorophore_type="Tetramethylrhodamine(TRITC)", psf_wavelength=5.78e-07)

        depth = 0.0
        radial = numpy.arange(0.0, self.radial_cutoff, self.radial_resolution, dtype=float)
        psf_r = psf.get_distribution(radial, depth)
        self.assertIs(type(psf_r), numpy.ndarray)
        self.assertEqual(psf_r.ndim, 1)
        self.assertEqual(psf_r.size, radial.size)
        self.assertTrue((psf_r >= 0.0).all())

        tot_r = numpy.sum(2 * numpy.pi * radial * psf_r) * self.radial_resolution
        print(f'Integral of radial distribution = {tot_r}')

        psf_cart = psf.radial_to_cartesian(radial, psf_r, self.radial_cutoff, self.radial_resolution)
        tot_cart = psf_cart.sum() * (self.radial_resolution * self.radial_resolution)
        print(f'Integral of cartesian distribution = {tot_cart}')

        camera = numpy.zeros((512, 512))
        pixel_length = 4.444444444444444e-08
        psf.overlay_signal_(camera, psf_cart, numpy.zeros(3, dtype=float), pixel_length, self.radial_resolution, 1.0)
        # tot_camera = camera.sum() * (self.radial_resolution * self.radial_resolution)
        tot_camera = camera.sum()
        print(f'Integral of detected = {tot_camera}')

    def test3(self):
        print('Testing Gaussian ...')
        import numpy
        from scopyon._epifm import PointSpreadingFunction
        psf = PointSpreadingFunction(psf_radial_cutoff=self.radial_cutoff, psf_radial_width=1.0e-7, psf_depth_cutoff=1000.0e-9, fluorophore_type="Gaussian", psf_wavelength=6.0e-7)

        depth = 0.0
        radial = numpy.arange(0.0, self.radial_cutoff, self.radial_resolution, dtype=float)
        psf_r = psf.get_distribution(radial, depth)
        self.assertIs(type(psf_r), numpy.ndarray)
        self.assertEqual(psf_r.ndim, 1)
        self.assertEqual(psf_r.size, radial.size)
        self.assertTrue((psf_r >= 0.0).all())

        tot_r = numpy.sum(2 * numpy.pi * radial * psf_r) * self.radial_resolution
        print(f'Integral of radial distribution = {tot_r}')

        psf_cart = psf.radial_to_cartesian(radial, psf_r, self.radial_cutoff, self.radial_resolution)
        tot_cart = psf_cart.sum() * (self.radial_resolution * self.radial_resolution)
        print(f'Integral of cartesian distribution = {tot_cart}')

        camera = numpy.zeros((512, 512))
        pixel_length = 4.444444444444444e-08
        psf.overlay_signal_(camera, psf_cart, numpy.zeros(3, dtype=float), pixel_length, self.radial_resolution, 1.0)
        # tot_camera = camera.sum() * (self.radial_resolution * self.radial_resolution)
        tot_camera = camera.sum()
        print(f'Integral of detected = {tot_camera}')


if __name__ == '__main__':
    unittest.main()
