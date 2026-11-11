"""Provide utility functions for map geometry and key point calculations."""

from point import Point


def _eval_absolute_map_points(blue_base: Point, red_base: Point) -> dict[str, Point]:
    r"""Computes key points of the map, based on position of both bases.
    
    Approximate map layout, with key points marked:
        bb                yaxis <--
          \                        |
            bm - br                v
            |    |               xaxis
            bl - m  - rr
                 |    |
                 rl - rm
                        \
                         rb
    legend:
        bb - blue_base
        bm - blue_mid
        bl - blue_left
        br - blue_right
        m - mid
        rl - red_left
        rr - red_right
        rm - red_mid
        rb - red_base

    Args:
        blue_base: Position of the blue team's base.
        red_base: Position of the red team's base.

    Returns:
        A dictionary mapping point names to their coordinates.
        Keys are full point team-indepentend names (e.g., "blue_left", "red_mid").
        Values are Point objects.
    """
    mid = {
        "x": (blue_base["x"] + red_base["x"]) / 2.0,
        "y": (blue_base["y"] + red_base["y"]) / 2.0,
    }
    diff = abs((mid["x"] - blue_base["x"]) * 9.0 / 10.5)
    blue_mid = {"x": mid["x"] - diff, "y": mid["y"] + diff}
    blue_left = {"x": mid["x"], "y": blue_mid["y"]}
    blue_right = {"x": blue_mid["x"], "y": mid["y"]}
    red_mid = {"x": mid["x"] + diff, "y": mid["y"] - diff}
    red_left = {"x": red_mid["x"], "y": mid["y"]}
    red_right = {"x": mid["x"], "y": red_mid["y"]}

    map_points: dict[str, Point] = {}
    map_points["blue_base"] = {"x": blue_base["x"], "y": blue_base["y"]}
    map_points["red_base"] = {"x": red_base["x"], "y": red_base["y"]}
    map_points["mid"] = mid
    map_points["blue_mid"] = blue_mid
    map_points["blue_left"] = blue_left
    map_points["blue_right"] = blue_right
    map_points["red_mid"] = red_mid
    map_points["red_left"] = red_left
    map_points["red_right"] = red_right

    return map_points


def _convert_to_team_relative(
    map_points: dict[str, Point],
) -> tuple[dict[str, Point], dict[str, Point]]:
    """Converts map points names to team_relative.

    Changes 'blue' and 'red' naming to the 'my' and 'enemy'.
    Swaps left and right for blue team perspective.

    Returns:
        A tuple containing (blue_perspective, red_perspective).
    """
    blue_perspective = {
        "my_base": map_points["blue_base"],
        "my_mid": map_points["blue_mid"],
        "my_left": map_points["blue_right"],
        "my_right": map_points["blue_left"],
        "mid": map_points["mid"],
        "enemy_left": map_points["red_right"],
        "enemy_right": map_points["red_left"],
        "enemy_mid": map_points["red_mid"],
        "enemy_base": map_points["red_base"],
    }
    red_perspective = {
        "my_base": map_points["red_base"],
        "my_mid": map_points["red_mid"],
        "my_left": map_points["red_left"],
        "my_right": map_points["red_right"],
        "mid": map_points["mid"],
        "enemy_left": map_points["blue_left"],
        "enemy_right": map_points["blue_right"],
        "enemy_mid": map_points["blue_mid"],
        "enemy_base": map_points["blue_base"],
    }
    return blue_perspective, red_perspective


def eval_map_points(blue_base: Point, red_base: Point) -> tuple[dict[str, Point], dict[str, Point]]:
    """Evaluate map points based on the positions of both team bases.

    Calculates absolute map points and converts them to team-relative
    naming for both blue and red perspectives.

    Args:
        blue_base: Position of the blue team's base.
        red_base: Position of the red team's base.

    Returns:
        A tuple containing (blue_perspective, red_perspective).
    """
    return _convert_to_team_relative(_eval_absolute_map_points(blue_base, red_base))
