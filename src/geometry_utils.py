"""Geometric primitives and potential field functions for 2D points."""

import math

from point import Point


def distance(a: Point, b: Point) -> float:
    """Computes the raw Euclidean distance ||b - a||_2 between two points.

    Args:
        a (Point): First 2D point.
        b (Point): Second 2D point.

    Returns:
        float: Euclidean distance between point a and point b.
    """
    return math.hypot(b["x"] - a["x"], b["y"] - a["y"])


def norm_dist(a: Point, b: Point, scale: float) -> float:
    r"""Computes normalized Euclidean distance \hat{d}(a, b) = ||b - a||_2 / L.

    Args:
        a (Point): First 2D point.
        b (Point): Second 2D point.
        scale (float): Normalization length scale L > 0.

    Returns:
        float: Scale-normalized distance.
    """
    return distance(a, b) / scale


def norm_seg_dist(p: Point, start: Point, end: Point, scale: float) -> float:
    r"""Computes normalized point-to-segment distance \hat{d}_{seg}(p, start, end).

    Args:
        p (Point): Query point (e.g., enemy position).
        start (Point): Segment start point (e.g., current position).
        end (Point): Segment end point (e.g., candidate destination).
        scale (float): Normalization length scale L > 0.

    Returns:
        float: Shortest normalized distance from p to the line segment start->end.
    """
    dx = end["x"] - start["x"]
    dy = end["y"] - start["y"]
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq == 0.0:
        return norm_dist(p, start, scale)

    t = max(
        0.0,
        min(1.0, ((p["x"] - start["x"]) * dx + (p["y"] - start["y"]) * dy) / seg_len_sq),
    )
    proj: Point = {"x": start["x"] + t * dx, "y": start["y"] + t * dy}
    return norm_dist(p, proj, scale)


def territory_polarity(candidate: Point, my_base: Point, enemy_base: Point) -> float:
    """Computes territory polarity T(c) in [-1.0, 1.0].

    Args:
        candidate (Point): Candidate point c being evaluated.
        my_base (Point): Position of the controlled player's base b_{my}.
        enemy_base (Point): Position of the enemy player's base b_{en}.

    Returns:
        float: Value in [-1.0, 1.0], where > 0 is enemy territory.
    """
    d_my = distance(candidate, my_base)
    d_en = distance(candidate, enemy_base)
    total = d_my + d_en
    return (d_my - d_en) / total if total > 0.0 else 0.0


def threat_repulsion(
    candidate: Point,
    my_pos: Point,
    enemy_pos: Point,
    scale: float,
    epsilon: float,
) -> float:
    r"""Computes the inverse-distance threat barrier \Phi(c).

    \Phi(c) = 1 / (\hat{d}_{seg}(p_{en}, p_{my}, c) + \epsilon)

    Args:
        candidate (Point): Candidate target point c.
        my_pos (Point): Controlled player's position p_{my}.
        enemy_pos (Point): Enemy player's position p_{en}.
        scale (float): Map length scale L.
        epsilon (float): Smoothing constant \epsilon > 0 preventing division by zero.

    Returns:
        float: Repulsion magnitude along the trajectory p_{my} -> c.
    """
    clearance = norm_seg_dist(enemy_pos, my_pos, candidate, scale)
    return 1.0 / (clearance + epsilon)
