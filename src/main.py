"""Main controller connecting OSC communication, map utilities, and AI."""

import logging
from typing import Callable

from map_utils import eval_map_points
from osc_messenger import OSCMessenger
from point import Point

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Type alias for AI function
AgentFn = Callable[[Point, Point, bool, bool, dict[str, Point]], Point]


def dummy_ai(
    my_pos: Point,
    enemy_pos: Point,
    my_carry_flag: bool,
    enemy_carry_flag: bool,
    my_map: dict[str, Point],
) -> Point:
    """Placeholder AI that captures the flag and returns to base.

    Args:
        my_pos (Point): Current position of the controlled player.
        enemy_pos (Point): Current position of the enemy player.
        my_carry_flag (bool): True if the controlled player carries the flag.
        enemy_carry_flag (bool): True if the enemy player carries the flag.
        my_map (dict[str, Point]): Team-relative map waypoints.

    Returns:
        Point: Target destination coordinates.
    """
    return my_map["my_base"] if my_carry_flag else my_map["enemy_base"]


def run_game_loop(
    p1_ai: AgentFn,
    p2_ai: AgentFn,
    listen_ip: str = "127.0.0.1",
    listen_port: int = 8000,
    unreal_ip: str = "127.0.0.1",
    unreal_port: int = 8001,
) -> None:
    """Initializes OSC and runs the receive -> compute -> send cycle.

    Args:
        p1_ai (AgentFn): Decision function for Player 1 (Blue).
        p2_ai (AgentFn): Decision function for Player 2 (Red).
        listen_ip (str): IP address for Python to listen on.
        listen_port (int): Port for Python to listen on.
        unreal_ip (str): IP address of the UE5 instance.
        unreal_port (int): Port of the UE5 instance.
    """
    osc = OSCMessenger(
        listen_ip=listen_ip,
        listen_port=listen_port,
        unreal_ip=unreal_ip,
        unreal_port=unreal_port,
    )
    maps: dict[str, dict[str, Point]] = {}

    def on_init(packet_id: int, b1_x: float, b1_y: float, b2_x: float, b2_y: float) -> None:
        """Calculates relative map points and replies 'ok'."""
        b1: Point = {"x": float(b1_x), "y": float(b1_y)}
        b2: Point = {"x": float(b2_x), "y": float(b2_y)}
        maps["p1"], maps["p2"] = eval_map_points(b1, b2)
        logger.info("Initialized match (packet_id=%d): B1=%s, B2=%s", packet_id, b1, b2)
        osc.send_ok(packet_id)

    def on_step(
        packet_id: int,
        p1_x: float,
        p1_y: float,
        p2_x: float,
        p2_y: float,
        p1_flag: bool,
        p2_flag: bool,
    ) -> None:
        """Calls both AI functions and sends destinations back to UE5."""
        if "p1" not in maps or "p2" not in maps:
            logger.warning("Ignored 'step' (packet_id=%d): match not initialized yet.", packet_id)
            return

        p1_pos: Point = {"x": float(p1_x), "y": float(p1_y)}
        p2_pos: Point = {"x": float(p2_x), "y": float(p2_y)}
        p1_has_flag = bool(p1_flag)
        p2_has_flag = bool(p2_flag)

        p1_dest = p1_ai(p1_pos, p2_pos, p1_has_flag, p2_has_flag, maps["p1"])
        p2_dest = p2_ai(p2_pos, p1_pos, p2_has_flag, p1_has_flag, maps["p2"])

        logger.debug("Step %d -> P1 dest: %s, P2 dest: %s", packet_id, p1_dest, p2_dest)
        osc.send_step(packet_id, p1_dest["x"], p1_dest["y"], p2_dest["x"], p2_dest["y"])

    def on_end(packet_id: int, winner_id: int) -> None:
        """Acknowledges match end and resets stored maps."""
        logger.info("Match ended (packet_id=%d). Winner Player ID: %d", packet_id, winner_id)
        maps.clear()
        osc.send_ok(packet_id)

    osc.register_handler("init", on_init)
    osc.register_handler("step", on_step)
    osc.register_handler("end", on_end)

    osc.start()


if __name__ == "__main__":
    run_game_loop(p1_ai=dummy_ai, p2_ai=dummy_ai)
