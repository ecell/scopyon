import unittest

import numpy


class TestEPIFM(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test1(self):
        import scopyon._epifm

    def test2(self):
        from scopyon._epifm import PointSpreadingFunction
        fluoem_norm = numpy.zeros(1, dtype=float)
        fluoem_norm[0] = 1.0  #XXX: Only the sum matters here
        psf = PointSpreadingFunction(psf_radial_cutoff=1000.0e-9, psf_radial_width=None, psf_depth_cutoff=1000.0e-9, fluoem_norm=fluoem_norm, dichroic_switch=False, dichroic_eff=None, emission_switch=False, emission_eff=None, fluorophore_type="Tetramethylrhodamine(TRITC)", psf_wavelength=5.78e-07, psf_normalization=1.0)

        depth = 0.0
        radial = numpy.arange(0.0, 1000.0e-9, 1.0e-9, dtype=float)
        psf_r = psf.get_distribution(radial, depth)
        self.assertIs(type(psf_r), numpy.ndarray)
        self.assertEqual(psf_r.ndim, 1)
        self.assertEqual(psf_r.size, radial.size)

    def test3(self):
        from scopyon._epifm import PointSpreadingFunction
        fluoem_norm = numpy.zeros(1, dtype=float)
        fluoem_norm[0] = 1.0  #XXX: Only the sum matters here
        psf = PointSpreadingFunction(psf_radial_cutoff=1000.0e-9, psf_radial_width=1.0e-7, psf_depth_cutoff=1000.0e-9, fluoem_norm=fluoem_norm, dichroic_switch=False, dichroic_eff=None, emission_switch=False, emission_eff=None, fluorophore_type="Gaussian", psf_wavelength=6.0e-7, psf_normalization=1.0)

        depth = 0.0
        radial = numpy.arange(0.0, 1000.0e-9, 1.0e-9, dtype=float)
        psf_r = psf.get_distribution(radial, depth)
        self.assertIs(type(psf_r), numpy.ndarray)
        self.assertEqual(psf_r.ndim, 1)
        self.assertEqual(psf_r.size, radial.size)


if __name__ == '__main__':
    unittest.main()
