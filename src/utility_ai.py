"""Deterministic Utility Function AI mapped directly to its mathematical model."""

from dataclasses import dataclass

from geometry_utils import (
    distance,
    norm_dist,
    norm_seg_dist,
    territory_polarity,
    threat_repulsion,
)
from point import Point


@dataclass(frozen=True)
class UtilityParams:
    r"""Weights and hyperparameters of the utility function U(c | S).

    Attributes:
        w_goal (float): Attraction weight toward the primary base objective.
        w_travel (float): Penalty weight for step travel distance.
        w_threat (float): Repulsion weight from enemy trajectories.
        w_int (float): Attraction weight for intercepting an enemy carrier.
        w_terr (float): Bias weight toward enemy territory.
        alpha (float): Scaling factor for the threat barrier \Phi(c).
        beta (float): Catch-up ratio threshold for chasing vs. counter-capturing.
        gamma (float): Blend factor between direct pursuit and path cutoff in [0, 1].
        epsilon (float): Smoothing constant for threat repulsion barrier.
        r_arr (float): Arrival radius to filter out already-reached intermediate nodes.
    """

    w_goal: float = 2.0
    w_travel: float = 0.5
    w_threat: float = 3.0
    w_int: float = 2.5
    w_terr: float = 0.4
    alpha: float = 0.15
    beta: float = 1.15
    gamma: float = 0.6
    epsilon: float = 0.1
    r_arr: float = 50.0


DEFAULT_PARAMS = UtilityParams()


def u_base(
    c: Point,
    p_my: Point,
    b_my: Point,
    b_en: Point,
    scale: float,
    params: UtilityParams,
) -> float:
    r"""Computes state-independent baseline utility U_{base}(c).

    U_{base}(c) = w_{terr} * T(c) - w_{travel} * \hat{d}(p_{my}, c)

    Args:
        c (Point): Candidate point c.
        p_my (Point): Controlled player's position p_{my}.
        b_my (Point): Controlled player's base b_{my}.
        b_en (Point): Enemy player's base b_{en}.
        scale (float): Map normalization scale L.
        params (UtilityParams): Model parameters.

    Returns:
        float: Baseline utility value.
    """
    return params.w_terr * territory_polarity(c, b_my, b_en) - params.w_travel * norm_dist(
        p_my, c, scale
    )


def u_returning(
    c: Point,
    p_my: Point,
    p_en: Point,
    b_my: Point,
    scale: float,
    params: UtilityParams,
) -> float:
    r"""Computes flag-return utility with threat avoidance.

    U_{ret}(c) = -w_{goal} * \hat{d}(c, b_{my}) - w_{threat} * \alpha * \Phi(c)

    Args:
        c (Point): Candidate point c.
        p_my (Point): Controlled player's position p_{my}.
        p_en (Point): Enemy player's position p_{en}.
        b_my (Point): Controlled player's base b_{my}.
        scale (float): Map normalization scale L.
        params (UtilityParams): Model parameters.

    Returns:
        float: Tactical utility when returning home with the flag.
    """
    phi = threat_repulsion(c, p_my, p_en, scale, params.epsilon)
    return -params.w_goal * norm_dist(c, b_my, scale) - params.w_threat * params.alpha * phi


def u_defending(
    c: Point,
    p_my: Point,
    p_en: Point,
    b_en: Point,
    scale: float,
    params: UtilityParams,
) -> float:
    """Computes defending sub-function U_{def}(c) when only the enemy has a flag.

    Args:
        c (Point): Candidate point c.
        p_my (Point): Controlled player's position p_{my}.
        p_en (Point): Enemy player's position p_{en}.
        b_en (Point): Enemy player's base b_{en}.
        scale (float): Map normalization scale L.
        params (UtilityParams): Model parameters.

    Returns:
        float: Tactical utility for intercepting or counter-capturing.
    """
    if norm_dist(p_my, p_en, scale) <= params.beta * norm_dist(p_en, b_en, scale):
        pursuit = norm_dist(c, p_en, scale)
        cutoff = norm_seg_dist(c, p_en, b_en, scale)
        return -params.w_int * (params.gamma * pursuit + (1.0 - params.gamma) * cutoff)
    return -params.w_goal * norm_dist(c, b_en, scale)


def u_standoff(
    c: Point,
    p_my: Point,
    p_en: Point,
    b_my: Point,
    b_en: Point,
    scale: float,
    params: UtilityParams,
) -> float:
    """Computes standoff sub-function U_{standoff}(c) when both players carry flags.

    Args:
        c (Point): Candidate point c.
        p_my (Point): Controlled player's position p_{my}.
        p_en (Point): Enemy player's position p_{en}.
        b_my (Point): Controlled player's base b_{my}.
        b_en (Point): Enemy player's base b_{en}.
        scale (float): Map normalization scale L.
        params (UtilityParams): Model parameters.

    Returns:
        float: Tactical utility for evading to score or forcing a mutual reset.
    """
    if norm_dist(p_my, b_my, scale) <= norm_dist(p_en, b_en, scale):
        return u_returning(c, p_my, p_en, b_my, scale, params)
    return -params.w_int * norm_dist(c, p_en, scale)


def u_tactical(
    c: Point,
    p_my: Point,
    p_en: Point,
    f_my: bool,
    f_en: bool,
    b_my: Point,
    b_en: Point,
    scale: float,
    params: UtilityParams,
) -> float:
    """Evaluates the piecewise tactical utility U_{tactical}(c | S).

    Args:
        c (Point): Candidate point c.
        p_my (Point): Controlled player's position p_{my}.
        p_en (Point): Enemy player's position p_{en}.
        f_my (bool): True if controlled player carries a flag.
        f_en (bool): True if enemy player carries a flag.
        b_my (Point): Controlled player's base b_{my}.
        b_en (Point): Enemy player's base b_{en}.
        scale (float): Map normalization scale L.
        params (UtilityParams): Model parameters.

    Returns:
        float: Piecewise tactical utility score.
    """
    if not f_my and not f_en:
        return -params.w_goal * norm_dist(c, b_en, scale)
    if f_my and not f_en:
        return u_returning(c, p_my, p_en, b_my, scale, params)
    if not f_my and f_en:
        return u_defending(c, p_my, p_en, b_en, scale, params)
    return u_standoff(c, p_my, p_en, b_my, b_en, scale, params)


def u_total(
    c: Point,
    p_my: Point,
    p_en: Point,
    f_my: bool,
    f_en: bool,
    my_map: dict[str, Point],
    params: UtilityParams = DEFAULT_PARAMS,
) -> float:
    """Computes total utility U(c | S) = U_{base}(c) + U_{tactical}(c | S).

    Args:
        c (Point): Candidate destination point c.
        p_my (Point): Controlled player's position p_{my}.
        p_en (Point): Enemy player's position p_{en}.
        f_my (bool): True if controlled player carries a flag.
        f_en (bool): True if enemy player carries a flag.
        my_map (dict[str, Point]): Team-relative map waypoints M.
        params (UtilityParams): Model parameters.

    Returns:
        float: Total scalar utility U(c | S).
    """
    b_my = my_map["my_base"]
    b_en = my_map["enemy_base"]
    scale = max(distance(b_my, b_en), 1.0)

    return u_base(c, p_my, b_my, b_en, scale, params) + u_tactical(
        c, p_my, p_en, f_my, f_en, b_my, b_en, scale, params
    )


def build_candidate_set(
    p_my: Point,
    p_en: Point,
    f_my: bool,
    f_en: bool,
    my_map: dict[str, Point],
    r_arr: float,
) -> list[Point]:
    r"""Constructs the valid candidate set C \subseteq M \cup {p_{en}}.

    Args:
        p_my (Point): Controlled player's position p_{my}.
        p_en (Point): Enemy player's position p_{en}.
        f_my (bool): True if controlled player carries a flag.
        f_en (bool): True if enemy player carries a flag.
        my_map (dict[str, Point]): Team-relative map waypoints M.
        r_arr (float): Arrival radius threshold.

    Returns:
        list[Point]: Filtered list of candidate destination points.
    """
    candidates: list[Point] = []
    for name, pt in my_map.items():
        if name == "my_base" and not f_my:
            continue
        is_goal = (name == "my_base" and f_my) or (name == "enemy_base" and not f_my)
        if not is_goal and distance(p_my, pt) < r_arr:
            continue
        candidates.append(pt)

    if f_en:
        candidates.append(p_en)

    return candidates


def utility_agent(
    my_pos: Point,
    enemy_pos: Point,
    my_carry_flag: bool,
    enemy_carry_flag: bool,
    my_map: dict[str, Point],
) -> Point:
    r"""Selects c^* = argmax_{c in C} U(c | S) for the current game step.

    Args:
        my_pos (Point): Controlled player's position p_{my}.
        enemy_pos (Point): Enemy player's position p_{en}.
        my_carry_flag (bool): Flag state f_{my}.
        enemy_carry_flag (bool): Flag state f_{en}.
        my_map (dict[str, Point]): Team-relative map waypoints M.

    Returns:
        Point: Optimal destination point c^*.
    """
    candidates = build_candidate_set(
        my_pos,
        enemy_pos,
        my_carry_flag,
        enemy_carry_flag,
        my_map,
        DEFAULT_PARAMS.r_arr,
    )
    return max(
        candidates,
        key=lambda c: u_total(c, my_pos, enemy_pos, my_carry_flag, enemy_carry_flag, my_map),
    )
