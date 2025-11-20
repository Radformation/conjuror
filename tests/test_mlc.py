from unittest import TestCase

import numpy as np

from conjuror.plans.mlc import (
    MLCShaper,
    split_sacrifice_travel,
    next_sacrifice_shift,
    interpolate_control_points,
)


class TestMLCShaper(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # simplistic MLC setup
        cls.leaf_boundaries: tuple[float, ...] = tuple(
            np.arange(start=-200, stop=201, step=5).astype(float)
        )

    def test_init(self):
        MLCShaper(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )

    def test_num_leaves(self):
        shaper = MLCShaper(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        self.assertEqual(shaper.num_leaves, 160)

    def test_meterset_over_1(self):
        shaper = MLCShaper(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        with self.assertRaises(ValueError):
            shaper.add_strip(position_mm=-5, strip_width_mm=0, meterset_at_target=2)

    def test_sacrifice_without_transition_dose(self):
        shaper = MLCShaper(
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
        shaper = MLCShaper(
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
        shaper = MLCShaper(
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
        shaper = MLCShaper(
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
        shaper = MLCShaper(
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
        shaper = MLCShaper(
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
        shaper = MLCShaper(
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
        res = split_sacrifice_travel(distance=33, max_travel=20)
        self.assertCountEqual(res, [20, 13])
        res = split_sacrifice_travel(distance=11, max_travel=20)
        self.assertCountEqual(res, [11])
        res = split_sacrifice_travel(distance=66, max_travel=20)
        self.assertCountEqual(res, [20, 20, 20, 6])

    def test_as_control_points(self):
        shaper = MLCShaper(
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
        shaper = MLCShaper(
            leaf_y_positions=self.leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        shaper.add_strip(position_mm=-5, strip_width_mm=0, meterset_at_target=1)
        metersets = shaper.as_metersets()
        self.assertEqual(metersets, [0, 1])


class TestNextSacrificeShift(TestCase):
    def test_easy(self):
        target = next_sacrifice_shift(
            current_position_mm=0,
            travel_mm=5,
            x_width_mm=400,
            other_mlc_position=0,
            max_overtravel_mm=140,
        )
        self.assertEqual(target, -5)

    def test_toward_target_right(self):
        target = next_sacrifice_shift(
            current_position_mm=-5,
            travel_mm=50,
            x_width_mm=400,
            other_mlc_position=0,
            max_overtravel_mm=140,
        )
        self.assertEqual(target, 50)

    def test_toward_target_left(self):
        target = next_sacrifice_shift(
            current_position_mm=45,
            travel_mm=50,
            x_width_mm=400,
            other_mlc_position=0,
            max_overtravel_mm=140,
        )
        self.assertEqual(target, -50)

    def test_travel_too_large(self):
        with self.assertRaises(ValueError):
            next_sacrifice_shift(
                current_position_mm=0,
                travel_mm=200,
                x_width_mm=400,
                other_mlc_position=0,
                max_overtravel_mm=140,
            )

    def test_travel_can_be_over_max_overtravel_if_on_other_side(self):
        target = next_sacrifice_shift(
            current_position_mm=0,
            travel_mm=200,
            x_width_mm=400,
            other_mlc_position=100,
            max_overtravel_mm=140,
        )
        self.assertEqual(target, 200)

    def test_at_edge_of_width(self):
        target = next_sacrifice_shift(
            current_position_mm=-180,
            travel_mm=30,
            x_width_mm=400,
            other_mlc_position=-190,
            max_overtravel_mm=140,
        )
        self.assertEqual(target, 30)

        target = next_sacrifice_shift(
            current_position_mm=180,
            travel_mm=30,
            x_width_mm=400,
            other_mlc_position=190,
            max_overtravel_mm=140,
        )
        self.assertEqual(target, -30)

    def test_width_vs_overtravel(self):
        with self.assertRaises(ValueError):
            next_sacrifice_shift(
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
            interpolate_control_points(
                control_point_start=[0, 0, 0, 0, 0],
                control_point_end=[10, 10, 10, 10],
                interpolation_ratios=[0.5],
                sacrifice_chunks=[5],
                max_overtravel=140,
            )

    def test_no_interpolation(self):
        with self.assertRaises(ValueError):
            interpolate_control_points(
                control_point_start=[0, 0, 0, 0, 0],
                control_point_end=[10, 10, 10, 10, 10],
                interpolation_ratios=[],
                sacrifice_chunks=[5],
                max_overtravel=140,
            )

    def test_interpolate_simple(self):
        interp_cp = interpolate_control_points(
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
        interp_cp = interpolate_control_points(
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
            interpolate_control_points(
                control_point_start=[0, 0, 0, 0, 0, 0],
                control_point_end=[10, 10, 10, 10, 10, 10],
                interpolation_ratios=[0.5],
                sacrifice_chunks=[30],
                max_overtravel=20,
            )

    def test_interpolation_over_1_or_0(self):
        with self.assertRaises(ValueError):
            interpolate_control_points(
                control_point_start=[0, 0, 0, 0, 0, 0],
                control_point_end=[10, 10, 10, 10, 10, 10],
                interpolation_ratios=[1.5],
                sacrifice_chunks=[5],
                max_overtravel=140,
            )
        with self.assertRaises(ValueError):
            interpolate_control_points(
                control_point_start=[0, 0, 0, 0, 0, 0],
                control_point_end=[10, 10, 10, 10, 10, 10],
                interpolation_ratios=[-0.5],
                sacrifice_chunks=[5],
                max_overtravel=140,
            )
