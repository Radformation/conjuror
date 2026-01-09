import numpy as np
import pytest

from conjuror.plans.mlc import (
    MLCModulator,
    MLCShaper,
    Park,
    Strip,
    Rectangle,
    RectangleMode,
)
from conjuror.plans.truebeam import TrueBeamMachine


@pytest.fixture(scope="class")
def shaper():
    machine = TrueBeamMachine(False)
    return MLCShaper(
        machine.mlc_boundaries,
        machine.specs.max_mlc_position,
        machine.specs.max_mlc_overtravel,
    )


@pytest.fixture(scope="class")
def leaf_boundaries():
    return tuple(np.arange(start=-200, stop=201, step=5).astype(float))


class TestShapes:
    def test_park(self, shaper):
        park = Park()
        shape = shaper.get_shape(park)
        assert all(s == -shaper.max_mlc_position for s in shape[:60])
        assert all(s == shaper.max_mlc_position for s in shape[60:])

    @pytest.mark.parametrize("position,width", [(-1, 2), (0, 2), (1, 2), (0, 0)])
    def test_strip(self, shaper, position, width):
        x_min = position - width / 2
        x_max = position + width / 2
        shape = Strip(position=position, width=width)
        mlc = shaper.get_shape(shape)
        assert all(m == x_min for m in mlc[:60])
        assert all(m == x_max for m in mlc[60:])

    @pytest.mark.parametrize("x_min,x_max", [(-1, 2), (1, 2), (0, 0)])
    def test_strip_from_minmax(self, shaper, x_min, x_max):
        shape = Strip.from_minmax(x_min=x_min, x_max=x_max)
        mlc = shaper.get_shape(shape)
        assert all(m == x_min for m in mlc[:60])
        assert all(m == x_max for m in mlc[60:])

    def test_strip_error_if_min_larger_than_max(self):
        x_min, x_max = 1, 0
        with pytest.raises(ValueError):
            Strip.from_minmax(x_min, x_max)

    RECTANGLE_TEST_PARAM = [
        (-5, 5, RectangleMode.EXACT),
        (-4.9, 5.1, RectangleMode.ROUND),
        (-5.1, 5.1, RectangleMode.INWARD),
        (-4.9, 4.9, RectangleMode.OUTWARD),
        (-5, 5, RectangleMode.ROUND),
        (-5, 5, RectangleMode.INWARD),
        (-5, 5, RectangleMode.OUTWARD),
    ]

    @pytest.mark.parametrize("y_min,y_max,y_mode", RECTANGLE_TEST_PARAM)
    def test_rectangle(self, shaper, y_min, y_max, y_mode):
        x_min, x_max = -100, 100
        shape = Rectangle(x_min, x_max, y_min, y_max, y_mode, 0, 0)
        mlc = shaper.get_shape(shape)
        assert all(x == 0 for x in mlc[:29])
        assert all(x == x_min for x in mlc[29:31])
        assert all(x == 0 for x in mlc[31:60])
        assert all(x == 0 for x in mlc[60 : 29 + 60])
        assert all(x == x_max for x in mlc[29 + 60 : 31 + 60])
        assert all(x == 0 for x in mlc[31 + 60 :])

    RECTANGLE_OPEN_TEST_PARAM = [
        RectangleMode.EXACT,
        RectangleMode.ROUND,
        RectangleMode.INWARD,
        RectangleMode.OUTWARD,
    ]

    @pytest.mark.parametrize("mode", RECTANGLE_OPEN_TEST_PARAM)
    def test_rectangle_open(self, shaper, mode):
        x_min, x_max, y_min, y_max = -100, 100, -200, 200
        shape = Rectangle(x_min, x_max, y_min, y_max, mode, 0, 0)
        mlc = shaper.get_shape(shape)
        assert all(x == -100 for x in mlc[:60])
        assert all(x == 100 for x in mlc[60:])

    @pytest.mark.parametrize("x_min,x_max,y_min,y_max", [(2, 1, 0, 1), (0, 1, 2, 1)])
    def test_rectangle_error_if_min_larger_than_max(self, x_min, x_max, y_min, y_max):
        with pytest.raises(ValueError):
            Rectangle(x_min, x_max, y_min, y_max, RectangleMode.ROUND, 0, 0)

    @pytest.mark.parametrize("y_min,y_max", [(-0.1, 0), (0, 0.1)])
    def test_rectangle_error_if_exact_and_not_possible(self, shaper, y_min, y_max):
        shape = Rectangle(0, 1, y_min, y_max, RectangleMode.EXACT, 0, 0)
        with pytest.raises(ValueError):
            shaper.get_shape(shape)


class TestMLCModulator:
    def test_init(self, leaf_boundaries):
        MLCModulator(
            leaf_y_positions=leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )

    def test_num_leaves(self, leaf_boundaries):
        shaper = MLCModulator(
            leaf_y_positions=leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        assert shaper.num_leaves == 160

    def test_meterset_over_1(self, leaf_boundaries):
        shaper = MLCModulator(
            leaf_y_positions=leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        with pytest.raises(ValueError):
            shaper.add_strip(position_mm=-5, strip_width_mm=0, meterset_at_target=2)

    def test_sacrifice_without_transition_dose(self, leaf_boundaries):
        shaper = MLCModulator(
            leaf_y_positions=leaf_boundaries,
            max_mlc_position=400,
            max_overtravel_mm=140,
        )
        with pytest.raises(ValueError):
            shaper.add_strip(
                position_mm=-5,
                strip_width_mm=0,
                meterset_at_target=1,
                meterset_transition=0,
                sacrificial_distance_mm=50,
            )

    def test_initial_sacrificial_gap(self, leaf_boundaries):
        shaper = MLCModulator(
            leaf_y_positions=leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        shaper.add_strip(
            position_mm=-5,
            strip_width_mm=0,
            meterset_at_target=1,
            initial_sacrificial_gap_mm=10,
        )
        assert shaper.control_points[0][0] == -10

    def test_cant_add_sacrificial_gap_after_first_point(self, leaf_boundaries):
        shaper = MLCModulator(
            leaf_y_positions=leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        shaper.add_strip(
            position_mm=-5,
            strip_width_mm=0,
            meterset_at_target=0.2,
            initial_sacrificial_gap_mm=5,
        )
        with pytest.raises(ValueError) as exc_info:
            shaper.add_strip(
                position_mm=-5,
                strip_width_mm=0,
                meterset_at_target=0.2,
                initial_sacrificial_gap_mm=10,
            )
        assert "already control points" in str(exc_info.value)

    def test_cant_have_initial_sacrifice_and_transition_dose(self, leaf_boundaries):
        shaper = MLCModulator(
            leaf_y_positions=leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        with pytest.raises(ValueError):
            shaper.add_strip(
                position_mm=-5,
                strip_width_mm=0,
                meterset_at_target=0,
                meterset_transition=1,
                initial_sacrificial_gap_mm=5,
            )

    def test_cant_have_meterset_transition_for_first_control_point(
        self, leaf_boundaries
    ):
        shaper = MLCModulator(
            leaf_y_positions=leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        with pytest.raises(ValueError) as exc_info:
            shaper.add_strip(
                position_mm=-5,
                strip_width_mm=0,
                meterset_at_target=0,
                meterset_transition=1,
            )
        assert "Cannot have a transition" in str(exc_info.value)

    def test_cant_have_initial_sacrificial_gap_and_sacrificial_distance(
        self, leaf_boundaries
    ):
        shaper = MLCModulator(
            leaf_y_positions=leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        with pytest.raises(ValueError) as exc_info:
            shaper.add_strip(
                position_mm=-5,
                strip_width_mm=0,
                meterset_at_target=0.5,
                meterset_transition=0.1,
                sacrificial_distance_mm=5,
                initial_sacrificial_gap_mm=5,
            )
        assert "Cannot specify both" in str(exc_info.value)

    def test_cannot_have_sacrifical_gap_on_secondary_control_point(
        self, leaf_boundaries
    ):
        shaper = MLCModulator(
            leaf_y_positions=leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        shaper.add_strip(position_mm=-5, strip_width_mm=0, meterset_at_target=0.5)
        with pytest.raises(ValueError) as exc_info:
            shaper.add_strip(
                position_mm=-5,
                strip_width_mm=0,
                meterset_at_target=0.5,
                initial_sacrificial_gap_mm=10,
            )
        assert "already control points" in str(exc_info.value)

    def test_split_sacrifices(self):
        res = MLCModulator._split_sacrifice_travel(distance=33, max_travel=20)
        assert set(res) == {20, 13}
        res = MLCModulator._split_sacrifice_travel(distance=11, max_travel=20)
        assert res == [11]
        res = MLCModulator._split_sacrifice_travel(distance=66, max_travel=20)
        assert set(res) == {20, 20, 20, 6}

    def test_as_control_points(self, leaf_boundaries):
        shaper = MLCModulator(
            leaf_y_positions=leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        shaper.add_strip(position_mm=-5, strip_width_mm=0, meterset_at_target=1)
        cp = shaper.as_control_points()
        assert len(cp) == 2  # start and end positions given meterset increments
        assert cp[0][0] == -5

    def test_as_metersets(self, leaf_boundaries):
        shaper = MLCModulator(
            leaf_y_positions=leaf_boundaries,
            max_mlc_position=200,
            max_overtravel_mm=140,
        )
        shaper.add_strip(position_mm=-5, strip_width_mm=0, meterset_at_target=1)
        metersets = shaper.as_metersets()
        assert metersets == [0, 1]


class TestNextSacrificeShift:
    def test_easy(self):
        target = MLCModulator._next_sacrifice_shift(
            current_position_mm=0,
            travel_mm=5,
            x_width_mm=400,
            other_mlc_position=0,
            max_overtravel_mm=140,
        )
        assert target == -5

    def test_toward_target_right(self):
        target = MLCModulator._next_sacrifice_shift(
            current_position_mm=-5,
            travel_mm=50,
            x_width_mm=400,
            other_mlc_position=0,
            max_overtravel_mm=140,
        )
        assert target == 50

    def test_toward_target_left(self):
        target = MLCModulator._next_sacrifice_shift(
            current_position_mm=45,
            travel_mm=50,
            x_width_mm=400,
            other_mlc_position=0,
            max_overtravel_mm=140,
        )
        assert target == -50

    def test_travel_too_large(self):
        with pytest.raises(ValueError):
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
        assert target == 200

    def test_at_edge_of_width(self):
        target = MLCModulator._next_sacrifice_shift(
            current_position_mm=-180,
            travel_mm=30,
            x_width_mm=400,
            other_mlc_position=-190,
            max_overtravel_mm=140,
        )
        assert target == 30

        target = MLCModulator._next_sacrifice_shift(
            current_position_mm=180,
            travel_mm=30,
            x_width_mm=400,
            other_mlc_position=190,
            max_overtravel_mm=140,
        )
        assert target == -30

    def test_width_vs_overtravel(self):
        with pytest.raises(ValueError):
            MLCModulator._next_sacrifice_shift(
                current_position_mm=0,
                travel_mm=30,
                x_width_mm=100,
                other_mlc_position=-190,
                max_overtravel_mm=140,
            )


class TestInterpolateControlPoints:
    """For these tests, we use a simplified version of a 3-pair MLC. The first and last pair are the sacrificial leaves."""

    def test_control_point_lengths_mismatch(self):
        with pytest.raises(ValueError):
            MLCModulator._interpolate_control_points(
                control_point_start=[0, 0, 0, 0, 0],
                control_point_end=[10, 10, 10, 10],
                interpolation_ratios=[0.5],
                sacrifice_chunks=[5],
                max_overtravel=140,
            )

    def test_no_interpolation(self):
        with pytest.raises(ValueError):
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
        assert interp_cp == [[-1, 5, -1, -1, 5, -1]]

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
        assert interp_cp[0] == cp1
        cp2 = [2, 5, 2, 2, 5, 2]
        assert interp_cp[1] == cp2
        cp3 = [9, 7.5, 9, 9, 7.5, 9]
        assert interp_cp[2] == cp3

    def test_overtravel(self):
        # 30 is over the max overtravel of 20
        with pytest.raises(ValueError):
            MLCModulator._interpolate_control_points(
                control_point_start=[0, 0, 0, 0, 0, 0],
                control_point_end=[10, 10, 10, 10, 10, 10],
                interpolation_ratios=[0.5],
                sacrifice_chunks=[30],
                max_overtravel=20,
            )

    def test_interpolation_over_1_or_0(self):
        with pytest.raises(ValueError):
            MLCModulator._interpolate_control_points(
                control_point_start=[0, 0, 0, 0, 0, 0],
                control_point_end=[10, 10, 10, 10, 10, 10],
                interpolation_ratios=[1.5],
                sacrifice_chunks=[5],
                max_overtravel=140,
            )
        with pytest.raises(ValueError):
            MLCModulator._interpolate_control_points(
                control_point_start=[0, 0, 0, 0, 0, 0],
                control_point_end=[10, 10, 10, 10, 10, 10],
                interpolation_ratios=[-0.5],
                sacrifice_chunks=[5],
                max_overtravel=140,
            )
