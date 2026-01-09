import copy
import tempfile

import numpy as np
import numpy.testing
import pydicom
import pytest


from conjuror.images.layers import (
    ConstantLayer,
    PerfectFieldLayer,
    RandomNoiseLayer,
)

from conjuror.images.layers import (
    ArrayLayer,
    Layer,
    SlopeLayer,
    clip_add,
    even_round,
)
from conjuror.images.simulators import (
    Simulator,
    IMAGER_AS1200,
    Imager,
    IMAGER_AS500,
    IMAGER_AS1000,
)

np.random.seed(1234)  # noqa


class TestClipAdd:
    def test_clip_add_normal(self):
        image1 = np.zeros((10, 10), dtype=np.uint16)
        image2 = np.ones((10, 10), dtype=np.uint16)
        output = clip_add(image1, image2, dtype=np.uint16)
        assert output.dtype == np.uint16
        assert output.shape == image1.shape
        numpy.testing.assert_array_equal(
            output, image2
        )  # arrays are equal because image1 is zeros

    def test_clip_doesnt_flip_bit(self):
        image1 = np.zeros((10, 10), dtype=np.uint16)
        image1.fill(np.iinfo(np.uint16).max)  # set image to max value
        image2 = np.ones((10, 10), dtype=np.uint16)
        output = clip_add(image1, image2, dtype=np.uint16)
        # adding 1 to an array at the max would normally flip bits; ensure it doesn't
        assert output.dtype == np.uint16
        numpy.testing.assert_array_equal(
            output, image1
        )  # output is same as image1 because image2 didn't actually add anything


class TestEvenRound:
    def test_even_round(self):
        assert even_round(3) == 4
        assert even_round(2) == 2
        assert even_round(15) == 16


class TestSlopeLayer:
    def test_slope_x(self):
        as1200 = Simulator(IMAGER_AS1200, sid=1000)
        # create field that will cover the whole image; want a uniform field
        as1200.add_layer(PerfectFieldLayer(field_size_mm=(400, 400), alpha=0.6))
        as1200.add_layer(SlopeLayer(slope_x=0.1, slope_y=0))
        # test that the left is less than the right edge
        left = as1200.image[:, 100].max()
        right = as1200.image[:, -100].max()
        assert left < right

    def test_negative_slope_x(self):
        as1200 = Simulator(IMAGER_AS1200, sid=1000)
        # create field that will cover the whole image; want a uniform field
        as1200.add_layer(PerfectFieldLayer(field_size_mm=(400, 400), alpha=0.6))
        as1200.add_layer(SlopeLayer(slope_x=-0.1, slope_y=0))
        # test that the left is greater than the right edge
        left = as1200.image[:, 100].max()
        right = as1200.image[:, -100].max()
        assert left > right

    def test_slope_y(self):
        as1200 = Simulator(IMAGER_AS1200, sid=1000)
        # create field that will cover the whole image; want a uniform field
        as1200.add_layer(PerfectFieldLayer(field_size_mm=(400, 400), alpha=0.6))
        as1200.add_layer(SlopeLayer(slope_x=0, slope_y=0.1))
        # test that the top is less than the bottom edge
        top = as1200.image[100, :].max()
        bottom = as1200.image[-100, :].max()
        assert top < bottom


class TestRandomNoise:
    def test_mean_doesnt_change(self):
        as1200 = Simulator(IMAGER_AS1200, sid=1000)
        as1200.add_layer(ConstantLayer(constant=35000))
        as1200.add_layer(RandomNoiseLayer(mean=0, sigma=0.001))
        assert as1200.image.mean() == pytest.approx(35000, abs=1)

    def test_std(self):
        as1200 = Simulator(IMAGER_AS1200, sid=1000)
        as1200.add_layer(ConstantLayer(constant=35000))
        as1200.add_layer(RandomNoiseLayer(mean=0, sigma=0.003))
        std = np.iinfo(np.uint16).max * 0.003
        assert as1200.image.std() == pytest.approx(std, abs=1)


class TestConstantLayer:
    def test_constant(self):
        as1200 = Simulator(IMAGER_AS1200, sid=1000)
        as1200.add_layer(ConstantLayer(constant=35))
        assert as1200.image.max() == pytest.approx(35, abs=1e-6)
        assert as1200.image.min() == pytest.approx(35, abs=1e-6)

    def test_two_constants(self):
        as1200 = Simulator(IMAGER_AS1200, sid=1000)
        as1200.add_layer(ConstantLayer(constant=35))
        as1200.add_layer(ConstantLayer(constant=11))
        assert as1200.image.max() == pytest.approx(46, abs=1e-6)
        assert as1200.image.min() == pytest.approx(46, abs=1e-6)

    def test_constant_wont_flip_bits_over(self):
        as1200 = Simulator(IMAGER_AS1200, sid=1000)
        as1200.add_layer(ConstantLayer(constant=35))
        as1200.add_layer(ConstantLayer(constant=777777777))
        assert as1200.image.max() == pytest.approx(np.iinfo(np.uint16).max, abs=1e-6)
        assert as1200.image.min() == pytest.approx(np.iinfo(np.uint16).max, abs=1e-6)

    def test_constant_wont_flip_bits_under(self):
        as1200 = Simulator(IMAGER_AS1200, sid=1000)
        as1200.add_layer(ConstantLayer(constant=35))
        as1200.add_layer(ConstantLayer(constant=-777777777))
        assert as1200.image.max() == pytest.approx(np.iinfo(np.uint16).min, abs=1e-6)
        assert as1200.image.min() == pytest.approx(np.iinfo(np.uint16).min, abs=1e-6)


class NOOPLayer(Layer):
    def apply(
        self, image: np.ndarray, pixel_size: float, mag_factor: float
    ) -> np.ndarray:
        return image


class TestArrayLayer:
    def test_array_layer_same_size(self):
        as1200 = Simulator(IMAGER_AS1200, sid=1000)
        as1200.add_layer(
            ArrayLayer(np.ones((as1200.image.shape[0], as1200.image.shape[1])))
        )
        assert np.all(as1200.image == 1)

    def test_smaller_size(self):
        as1200 = Simulator(IMAGER_AS1200, sid=1000)
        original_shape = as1200.image.shape
        as1200.add_layer(ArrayLayer(np.ones((5, 5))))
        assert np.max(as1200.image) == 1
        assert np.min(as1200.image) == 0
        # test center is 1
        assert as1200.image[as1200.image.shape[0] // 2, as1200.image.shape[1] // 2] == 1
        assert as1200.image.shape == original_shape

    def test_bigger_size(self):
        as1200 = Simulator(IMAGER_AS1200, sid=1000)
        original_shape = as1200.image.shape
        as1200.add_layer(
            ArrayLayer(np.ones((as1200.image.shape[0] + 5, as1200.image.shape[1] + 5)))
        )
        assert np.max(as1200.image) == 1
        # test shape didn't change even though the added array was bigger
        assert as1200.image.shape == original_shape


class SimulatorTestMixin:
    imager: Imager
    pixel_size: float
    shape: tuple[int, int]
    mag_factor = 1.5

    @pytest.fixture(autouse=True)
    def simulator(self):
        self.simulator = Simulator(self.imager)
        yield
        # Cleanup if needed

    def test_pixel_size(self):
        assert self.simulator.imager.pixel_size == self.pixel_size

    def test_shape(self):
        assert self.simulator.imager.shape == self.shape

    def test_image(self):
        assert self.simulator.image.shape == self.shape
        assert self.simulator.image.dtype == np.uint16

    def test_mag_factor(self):
        assert self.simulator.mag_factor == self.mag_factor
        ssd1000 = Simulator(imager=self.imager, sid=1000)
        assert ssd1000.mag_factor == 1

    def test_noop_layer_doesnt_change_image(self):
        sim = self.simulator
        orig_img = copy.deepcopy(sim.image)
        sim.add_layer(NOOPLayer())
        numpy.testing.assert_array_equal(sim.image, orig_img)

    def test_save_dicom(self):
        sim = self.simulator
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            sim.generate_dicom(tf.name, gantry_angle=12, coll_angle=33, table_angle=5)
            # shouldn't raise
            ds = pydicom.dcmread(tf.name)
        assert ds.pixel_array.shape == self.shape
        assert ds.GantryAngle == 12
        assert ds.BeamLimitingDeviceAngle == 33
        assert ds.PatientSupportAngle == 5

    def test_invert_array(self):
        sim = self.simulator
        sim.add_layer(PerfectFieldLayer(field_size_mm=(100, 100)))
        ds = sim.as_dicom(invert_array=False)
        # when false, the array retains the same values
        # the corner should be lower than the center, where dose was delivered
        mid = sim.image.shape[0] // 2
        assert float(ds.pixel_array[mid, mid]) > float(ds.pixel_array[0, 0])


class TestAS500(SimulatorTestMixin):
    imager = IMAGER_AS500
    pixel_size = 0.78125
    shape = (384, 512)


class TestAS1000(SimulatorTestMixin):
    imager = IMAGER_AS1000
    pixel_size = 0.390625
    shape = (768, 1024)


class TestAS1200(SimulatorTestMixin):
    imager = IMAGER_AS1200
    pixel_size = 0.336
    shape = (1280, 1280)


class TestCustomSimulator(SimulatorTestMixin):
    pixel_size = 0.123
    shape = (500, 700)
    imager = Imager(pixel_size, shape)

    def test_save_dicom(self):
        sim = Simulator(imager=self.imager)
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            sim.generate_dicom(tf.name, gantry_angle=12, coll_angle=33, table_angle=5)

    def test_invert_array(self):
        pass  # method not implemented
