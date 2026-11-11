"""Automated behavioral and contract tests for CTF AI agents."""

import itertools
import math
import warnings

import pytest

from decisionTree import step_advanced_decision_tree, step_simple_decision_tree
from map_utils import eval_map_points
from point import Point
from utility_ai import utility_agent

AGENTS = [
    utility_agent,
    step_simple_decision_tree,
    step_advanced_decision_tree,
]


class TacticalWarning(UserWarning):
    """Warning raised when an agent exhibits questionable but non-fatal tactics."""


@pytest.fixture
def my_map() -> dict[str, Point]:
    """Provides a standard team-relative map dictionary for testing.

    Returns:
        dict[str, Point]: Blue team perspective map waypoints.
    """
    blue_base: Point = {"x": 0.0, "y": 1000.0}
    red_base: Point = {"x": 1000.0, "y": 0.0}
    blue_map, _ = eval_map_points(blue_base, red_base)
    return blue_map


def _step_towards(current: Point, target: Point, step_size: float) -> Point:
    """Moves a point a fixed distance toward a target point.

    Args:
        current (Point): Starting coordinates.
        target (Point): Destination coordinates.
        step_size (float): Maximum distance to move in one step.

    Returns:
        Point: New coordinates after moving toward target.
    """
    dx = target["x"] - current["x"]
    dy = target["y"] - current["y"]
    dist = math.hypot(dx, dy)
    if dist <= step_size or dist == 0.0:
        return {"x": float(target["x"]), "y": float(target["y"])}
    return {
        "x": current["x"] + (dx / dist) * step_size,
        "y": current["y"] + (dy / dist) * step_size,
    }


@pytest.mark.parametrize("agent_fn", AGENTS)
@pytest.mark.parametrize("my_flag,enemy_flag", list(itertools.product([False, True], repeat=2)))
def test_return_type_and_no_crashes(
    agent_fn, my_map: dict[str, Point], my_flag: bool, enemy_flag: bool
) -> None:
    """Verifies that the agent runs without errors and returns a valid Point.

    Args:
        agent_fn: The AI decision function under test.
        my_map (dict[str, Point]): Team-relative map waypoints.
        my_flag (bool): Whether the controlled player carries a flag.
        enemy_flag (bool): Whether the enemy player carries a flag.
    """
    my_pos = my_map["my_left"]
    enemy_pos = my_map["enemy_left"]

    dest = agent_fn(my_pos, enemy_pos, my_flag, enemy_flag, my_map)

    assert isinstance(dest, dict), "Agent must return a Point dictionary."
    assert "x" in dest and "y" in dest, "Returned Point must contain 'x' and 'y' keys."
    assert isinstance(dest["x"], (int, float)) and isinstance(dest["y"], (int, float)), (
        "Point coordinates must be numeric."
    )


@pytest.mark.parametrize("agent_fn", AGENTS)
@pytest.mark.parametrize("enemy_flag", [False, True])
def test_rule_3_no_own_base_without_flag(
    agent_fn, my_map: dict[str, Point], enemy_flag: bool
) -> None:
    """Verifies if AI doesn't enter their own base without a flag.

    Args:
        agent_fn: The AI decision function under test.
        my_map (dict[str, Point]): Team-relative map waypoints.
        enemy_flag (bool): Whether the enemy player carries a flag.
    """
    for name, start_pos in my_map.items():
        if name == "my_base":
            continue
        dest = agent_fn(start_pos, my_map["mid"], False, enemy_flag, my_map)
        assert dest != my_map["my_base"], (
            f"Rule 3 violation: {agent_fn.__name__} targeted 'my_base' "
            f"from '{name}' without carrying a flag."
        )


@pytest.mark.parametrize("agent_fn", AGENTS)
def test_no_waypoint_oscillation_when_returning_flag(agent_fn, my_map: dict[str, Point]) -> None:
    """Warns if the agent oscillates between waypoints or stalls when returning a flag.

    Args:
        agent_fn: The AI decision function under test.
        my_map (dict[str, Point]): Team-relative map waypoints.
    """
    my_pos = dict(my_map["enemy_left"])
    enemy_pos = dict(my_map["enemy_mid"])
    step_size = 25.0
    max_steps = 150

    visited_positions: list[tuple[float, float]] = []

    for _ in range(max_steps):
        if (
            math.hypot(
                my_pos["x"] - my_map["my_base"]["x"],
                my_pos["y"] - my_map["my_base"]["y"],
            )
            < step_size
        ):
            return

        dest = agent_fn(my_pos, enemy_pos, True, False, my_map)
        my_pos = _step_towards(my_pos, dest, step_size)

        rounded_pos = (round(my_pos["x"], 1), round(my_pos["y"], 1))
        if visited_positions.count(rounded_pos) >= 3:
            warnings.warn(
                f"{agent_fn.__name__} oscillated around {rounded_pos} (last destination: {dest}).",
                TacticalWarning,
            )
            return
        visited_positions.append(rounded_pos)

    warnings.warn(
        f"{agent_fn.__name__} did not reach my_base within {max_steps} steps "
        f"against a stationary enemy.",
        TacticalWarning,
    )


@pytest.mark.parametrize("agent_fn", AGENTS)
def test_standoff_scores_when_adjacent_to_base(agent_fn, my_map: dict[str, Point]) -> None:
    """Warns if an agent adjacent to its base ignores scoring during a standoff.

    Args:
        agent_fn: The AI decision function under test.
        my_map (dict[str, Point]): Team-relative map waypoints.
    """
    my_pos = my_map["my_mid"]
    enemy_pos = my_map["enemy_mid"]

    dest = agent_fn(my_pos, enemy_pos, True, True, my_map)
    if dest != my_map["my_base"]:
        warnings.warn(
            f"{agent_fn.__name__} passed up scoring at 'my_base' from 'my_mid' "
            f"during a standoff, targeting {dest} instead.",
            TacticalWarning,
        )
