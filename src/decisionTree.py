"""Provides decision-tree based strategies and helper functions for CTF agents."""

import math
from typing import Literal

from point import Point


def _distance(p1: Point, p2: Point) -> float:
    return math.dist((p1["x"], p1["y"]), (p2["x"], p2["y"]))


def _is_closer(
    pos1: Point,
    pos2: Point,
    dest: Point,
    required_difference: float = 0.0,
) -> bool:
    return _distance(pos1, dest) + required_difference <= _distance(pos2, dest)


def _side_of_map(pos1: Point, map_points: dict[str, Point]) -> Literal["left", "right"]:
    """Determine which side of the map a position belongs to.

    Returns:
        "left" or "right", depending on the closer reference point.
    """
    if _is_closer(map_points["my_left"], map_points["my_right"], pos1):
        return "left"
    else:
        return "right"


def _is_same_side(pos1: Point, pos2: Point, map_points: dict[str, Point]) -> bool:
    return _side_of_map(pos1, map_points) == _side_of_map(pos2, map_points)


def step_simple_decision_tree(
    my_pos: Point,
    enemy_pos: Point,
    i_carry_flag: bool,
    enemy_carry_flag: bool,
    map_points: dict[str, Point],
):
    """Determine the next target position using a simple decision tree.

    Decision priorities:
        1. Rescuing allied flag if it's being held by enemy.
        2. Carrying enemy flag if the agent has it.
        3. Otherwise attack enemy base.

    Args:
        my_pos: Current position of the agent.
        enemy_pos: Current position of the enemy.
        i_carry_flag: Whether the agent is carrying the enemy flag.
        enemy_carry_flag: Whether the enemy is carrying the allied flag.
        map_points: Dictionary containing the map's key points.

    Returns:
        Target point for the agent's next movement.
    """
    if enemy_carry_flag:
        return enemy_pos
    if i_carry_flag:
        return map_points["my_base"]
    else:
        return map_points["enemy_base"]


def step_advanced_decision_tree(
    my_pos: Point,
    enemy_pos: Point,
    i_carry_flag: bool,
    enemy_carry_flag: bool,
    map_points: dict[str, Point],
):
    """Determine the next target position using a more advanced decision tree.

    It has simmilar fundamentals to the simple decision tree,
    but it has better navigation around the map.
    It tries to block path of the enemy that carries flag.
    When agent carries flag it tries to avoid enemy player.

    Decision priorities:
        1. Rescuing allied flag if it's being held by enemy.
        2. Carrying enemy flag if the agent has it.
        3. Otherwise attack enemy base.

    Args:
        my_pos: Current position of the agent.
        enemy_pos: Current position of the enemy.
        i_carry_flag: Whether the agent is carrying the enemy flag.
        enemy_carry_flag: Whether the enemy is carrying the allied flag.
        map_points: Dictionary containing the map's key points.

    Returns:
        Target point for the agent's next movement.
    """
    if enemy_carry_flag:
        if _is_closer(my_pos, enemy_pos, map_points["my_base"], required_difference=2):
            if _is_same_side(my_pos, enemy_pos, map_points):
                return enemy_pos
            else:
                if _is_closer(map_points["my_base"], map_points["enemy_base"], my_pos):
                    return map_points["mid"]
                else:
                    return map_points["enemy_mid"]
        else:
            return map_points["enemy_mid"]
    else:
        if i_carry_flag:
            if _is_closer(my_pos, enemy_pos, map_points["my_base"], required_difference=2.0):
                return map_points["my_base"]
            else:
                if _is_closer(map_points["my_base"], map_points["enemy_base"], my_pos):
                    if _is_same_side(my_pos, enemy_pos, map_points):
                        if _side_of_map(my_pos, map_points) == "left":
                            if _is_closer(map_points["mid"], map_points["my_right"], my_pos):
                                return map_points["my_right"]
                            else:
                                return map_points["mid"]
                        else:
                            if _is_closer(map_points["mid"], map_points["my_left"], my_pos):
                                return map_points["my_left"]
                            else:
                                return map_points["mid"]
                    else:
                        return map_points["my_base"]
                else:
                    if _is_closer(map_points["my_base"], map_points["enemy_base"], enemy_pos):
                        if _is_closer(
                            my_pos,
                            enemy_pos,
                            map_points["mid"],
                            required_difference=2.0,
                        ):
                            return map_points["mid"]
                        else:
                            if _side_of_map(my_pos, map_points) == "left":
                                return map_points["enemy_left"]
                            else:
                                return map_points["enemy_right"]
                    else:
                        if _is_same_side(my_pos, enemy_pos, map_points):
                            if _side_of_map(my_pos, map_points) == "left":
                                if _is_closer(
                                    map_points["enemy_mid"],
                                    map_points["enemy_right"],
                                    my_pos,
                                ):
                                    return map_points["enemy_right"]
                                else:
                                    return map_points["enemy_mid"]
                            else:
                                if _is_closer(
                                    map_points["enemy_mid"],
                                    map_points["enemy_left"],
                                    my_pos,
                                ):
                                    return map_points["enemy_left"]
                                else:
                                    return map_points["enemy_mid"]
                        else:
                            return map_points["mid"]
        else:
            return map_points["enemy_base"]
