from unittest import TestCase
from parameterized import parameterized
import numpy as np

from conjuror.plans.mlc import (
    MLCModulator,
    MLCShaper,
    Park,
    Strip,
    Rectangle,
    RectangleMode,
)
from conjuror.plans.truebeam import TrueBeamMachine


class TestShapes(TestCase):
    shaper: MLCShaper

    @classmethod
    def setUpClass(cls) -> None:
        machine = TrueBeamMachine(False)
        cls.shaper = MLCShaper(
            machine.mlc_boundaries,
            machine.specs.max_mlc_position,
            machine.specs.max_mlc_overtravel,
        )

    def test_park(self):
        park = Park()
        shape = self.shaper.get_shape(park)
        self.assertTrue(all(s == -self.shaper.max_mlc_position for s in shape[:60]))
        self.assertTrue(all(s == self.shaper.max_mlc_position for s in shape[60:]))

    @parameterized.expand([(-1, 1), (-1, -1), (0, 0), (1, 1)])
    def test_strip(self, x_min, x_max):
        shape = Strip(x_min=x_min, x_max=x_max)
        mlc = self.shaper.get_shape(shape)
        self.assertTrue(all(m == x_min for m in mlc[:60]))
        self.assertTrue(all(m == x_max for m in mlc[60:]))

    def test_strip_error_if_min_larger_than_max(self):
        with self.assertRaises(ValueError):
            Strip(x_min=0, x_max=-1)

    RECTANGLE_TEST_PARAM = [
        (-5, 5, RectangleMode.EXACT),
        (-4.9, 5.1, RectangleMode.ROUND),
        (-5.1, 5.1, RectangleMode.INWARD),
        (-4.9, 4.9, RectangleMode.OUTWARD),
    ]

    @parameterized.expand(RECTANGLE_TEST_PARAM)
    def test_rectangle(self, y_min, y_max, y_mode):
        x_min, x_max = -100, 100
        shape = Rectangle(x_min, x_max, y_min, y_max, y_mode, 0, 0)
        mlc = self.shaper.get_shape(shape)
        self.assertTrue(all(x == 0 for x in mlc[:29]))
        self.assertTrue(all(x == x_min for x in mlc[29:31]))
        self.assertTrue(all(x == 0 for x in mlc[31:60]))
        self.assertTrue(all(x == 0 for x in mlc[60 : 29 + 60]))
        self.assertTrue(all(x == x_max for x in mlc[29 + 60 : 31 + 60]))
        self.assertTrue(all(x == 0 for x in mlc[31 + 60 :]))

    @parameterized.expand([(2, 1, 0, 1), (0, 1, 2, 1)])
    def test_rectangle_error_if_min_larger_than_max(self, x_min, x_max, y_min, y_max):
        with self.assertRaises(ValueError):
            Rectangle(x_min, x_max, y_min, y_max, RectangleMode.ROUND, 0, 0)

    @parameterized.expand([(-0.1, 0), (0, 0.1)])
    def test_rectangle_error_if_exact_and_not_possible(self, y_min, y_max):
        shape = Rectangle(0, 1, y_min, y_max, RectangleMode.EXACT, 0, 0)
        with self.assertRaises(ValueError):
            self.shaper.get_shape(shape)


class TestMLCModulator(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # simplistic MLC setup
        cls.leaf_boundaries: tuple[float, ...] = tuple(
            np.arange(start=-200, stop=201, step=5).astype(float)
        )

    def test_init(self):
        MLCModulator(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )

    def test_num_leaves(self):
        shaper = MLCModulator(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        self.assertEqual(shaper.num_leaves, 160)

    def test_meterset_over_1(self):
        shaper = MLCModulator(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        with self.assertRaises(ValueError):
            shaper.add_strip(position_mm=-5, strip_width_mm=0, meterset_at_target=2)

    def test_sacrifice_without_transition_dose(self):
        shaper = MLCModulator(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=400,
            max_overtravel_mm=140,
        )
        with self.assertRaises(ValueError):
            shaper.add_strip(
                position_mm=-5,
                strip_width_mm=0,
                meterset_at_target=1,
                meterset_transition=0,
                sacrificial_distance_mm=50,
            )

    def test_initial_sacrificial_gap(self):
        shaper = MLCModulator(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        shaper.add_strip(
            position_mm=-5,
            strip_width_mm=0,
            meterset_at_target=1,
            initial_sacrificial_gap_mm=10,
        )
        self.assertEqual(shaper.control_points[0][0], -10)

    def test_cant_add_sacrificial_gap_after_first_point(self):
        shaper = MLCModulator(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        shaper.add_strip(
            position_mm=-5,
            strip_width_mm=0,
            meterset_at_target=0.2,
            initial_sacrificial_gap_mm=5,
        )
        with self.assertRaises(ValueError) as context:
            shaper.add_strip(
                position_mm=-5,
                strip_width_mm=0,
                meterset_at_target=0.2,
                initial_sacrificial_gap_mm=10,
            )
        self.assertIn("already control points", str(context.exception))

    def test_cant_have_initial_sacrifice_and_transition_dose(self):
        shaper = MLCModulator(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        with self.assertRaises(ValueError):
            shaper.add_strip(
                position_mm=-5,
                strip_width_mm=0,
                meterset_at_target=0,
                meterset_transition=1,
                initial_sacrificial_gap_mm=5,
            )

    def test_cant_have_meterset_transition_for_first_control_point(self):
        shaper = MLCModulator(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        with self.assertRaises(ValueError) as context:
            shaper.add_strip(
                position_mm=-5,
                strip_width_mm=0,
                meterset_at_target=0,
                meterset_transition=1,
            )
        self.assertIn("Cannot have a transition", str(context.exception))

    def test_cant_have_initial_sacrificial_gap_and_sacrificial_distance(self):
        shaper = MLCModulator(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        with self.assertRaises(ValueError) as context:
            shaper.add_strip(
                position_mm=-5,
                strip_width_mm=0,
                meterset_at_target=0.5,
                meterset_transition=0.1,
                sacrificial_distance_mm=5,
                initial_sacrificial_gap_mm=5,
            )
        self.assertIn("Cannot specify both", str(context.exception))

    def test_cannot_have_sacrifical_gap_on_secondary_control_point(self):
        shaper = MLCModulator(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        shaper.add_strip(position_mm=-5, strip_width_mm=0, meterset_at_target=0.5)
        with self.assertRaises(ValueError) as context:
            shaper.add_strip(
                position_mm=-5,
                strip_width_mm=0,
                meterset_at_target=0.5,
                initial_sacrificial_gap_mm=10,
            )
        self.assertIn("already control points", str(context.exception))

    def test_split_sacrifices(self):
        res = MLCModulator._split_sacrifice_travel(distance=33, max_travel=20)
        self.assertCountEqual(res, [20, 13])
        res = MLCModulator._split_sacrifice_travel(distance=11, max_travel=20)
        self.assertCountEqual(res, [11])
        res = MLCModulator._split_sacrifice_travel(distance=66, max_travel=20)
        self.assertCountEqual(res, [20, 20, 20, 6])

    def test_as_control_points(self):
        shaper = MLCModulator(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        shaper.add_strip(position_mm=-5, strip_width_mm=0, meterset_at_target=1)
        cp = shaper.as_control_points()
        self.assertEqual(
            len(cp), 2
        )  # start and end positions given meterset increments
        self.assertEqual(cp[0][0], -5)

    def test_as_metersets(self):
        shaper = MLCModulator(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        shaper.add_strip(position_mm=-5, strip_width_mm=0, meterset_at_target=1)
        metersets = shaper.as_metersets()
        self.assertEqual(metersets, [0, 1])


class TestNextSacrificeShift(TestCase):
    def test_easy(self):
        target = MLCModulator._next_sacrifice_shift(
            current_position_mm=0,
            travel_mm=5,
            x_width_mm=400,
            other_mlc_position=0,
            max_overtravel_mm=140,
        )
        self.assertEqual(target, -5)

    def test_toward_target_right(self):
        target = MLCModulator._next_sacrifice_shift(
            current_position_mm=-5,
            travel_mm=50,
            x_width_mm=400,
            other_mlc_position=0,
            max_overtravel_mm=140,
        )
        self.assertEqual(target, 50)

    def test_toward_target_left(self):
        target = MLCModulator._next_sacrifice_shift(
            current_position_mm=45,
            travel_mm=50,
            x_width_mm=400,
            other_mlc_position=0,
            max_overtravel_mm=140,
        )
        self.assertEqual(target, -50)

    def test_travel_too_large(self):
        with self.assertRaises(ValueError):
            MLCModulator._next_sacrifice_shift(
                current_position_mm=0,
                travel_mm=200,
                x_width_mm=400,
                other_mlc_position=0,
                max_overtravel_mm=140,
            )

    def test_travel_can_be_over_max_overtravel_if_on_other_side(self):
        target = MLCModulator._next_sacrifice_shift(
            current_position_mm=0,
            travel_mm=200,
            x_width_mm=400,
            other_mlc_position=100,
            max_overtravel_mm=140,
        )
        self.assertEqual(target, 200)

    def test_at_edge_of_width(self):
        target = MLCModulator._next_sacrifice_shift(
            current_position_mm=-180,
            travel_mm=30,
            x_width_mm=400,
            other_mlc_position=-190,
            max_overtravel_mm=140,
        )
        self.assertEqual(target, 30)

        target = MLCModulator._next_sacrifice_shift(
            current_position_mm=180,
            travel_mm=30,
            x_width_mm=400,
            other_mlc_position=190,
            max_overtravel_mm=140,
        )
        self.assertEqual(target, -30)

    def test_width_vs_overtravel(self):
        with self.assertRaises(ValueError):
            MLCModulator._next_sacrifice_shift(
                current_position_mm=0,
                travel_mm=30,
                x_width_mm=100,
                other_mlc_position=-190,
                max_overtravel_mm=140,
            )


class TestInterpolateControlPoints(TestCase):
    """For these tests, we use a simplified version of a 3-pair MLC. The first and last pair are the sacrificial leaves."""

    def test_control_point_lengths_mismatch(self):
        with self.assertRaises(ValueError):
            MLCModulator._interpolate_control_points(
                control_point_start=[0, 0, 0, 0, 0],
                control_point_end=[10, 10, 10, 10],
                interpolation_ratios=[0.5],
                sacrifice_chunks=[5],
                max_overtravel=140,
            )

    def test_no_interpolation(self):
        with self.assertRaises(ValueError):
            MLCModulator._interpolate_control_points(
                control_point_start=[0, 0, 0, 0, 0],
                control_point_end=[10, 10, 10, 10, 10],
                interpolation_ratios=[],
                sacrifice_chunks=[5],
                max_overtravel=140,
            )

    def test_interpolate_simple(self):
        interp_cp = MLCModulator._interpolate_control_points(
            control_point_start=[0, 0, 0, 0, 0, 0],
            control_point_end=[10, 10, 10, 10, 10, 10],
            interpolation_ratios=[0.5],
            sacrifice_chunks=[1],
            max_overtravel=140,
        )
        # the first, middle, and last values should be the sacrifice
        # the middle values should be the interpolated values
        self.assertEqual(interp_cp, [[-1, 5, -1, -1, 5, -1]])

    def test_interpolate_multiple(self):
        interp_cp = MLCModulator._interpolate_control_points(
            control_point_start=[0, 0, 0, 0, 0, 0],
            control_point_end=[10, 10, 10, 10, 10, 10],
            interpolation_ratios=[0.25, 0.5, 0.75],
            sacrifice_chunks=[3, 5, 7],
            max_overtravel=140,
        )
        # the sacrifice goes 0 - 3 -> -3 + 5 -> 2 + 7 -> 9
        cp1 = [-3, 2.5, -3, -3, 2.5, -3]
        self.assertEqual(interp_cp[0], cp1)
        cp2 = [2, 5, 2, 2, 5, 2]
        self.assertEqual(interp_cp[1], cp2)
        cp3 = [9, 7.5, 9, 9, 7.5, 9]
        self.assertEqual(interp_cp[2], cp3)

    def test_overtravel(self):
        # 30 is over the max overtravel of 20
        with self.assertRaises(ValueError):
            MLCModulator._interpolate_control_points(
                control_point_start=[0, 0, 0, 0, 0, 0],
                control_point_end=[10, 10, 10, 10, 10, 10],
                interpolation_ratios=[0.5],
                sacrifice_chunks=[30],
                max_overtravel=20,
            )

    def test_interpolation_over_1_or_0(self):
        with self.assertRaises(ValueError):
            MLCModulator._interpolate_control_points(
                control_point_start=[0, 0, 0, 0, 0, 0],
                control_point_end=[10, 10, 10, 10, 10, 10],
                interpolation_ratios=[1.5],
                sacrifice_chunks=[5],
                max_overtravel=140,
            )
        with self.assertRaises(ValueError):
            MLCModulator._interpolate_control_points(
                control_point_start=[0, 0, 0, 0, 0, 0],
                control_point_end=[10, 10, 10, 10, 10, 10],
                interpolation_ratios=[-0.5],
                sacrifice_chunks=[5],
                max_overtravel=140,
            )
