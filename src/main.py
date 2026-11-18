"""Main controller connecting OSC communication, map utilities, and AI."""

import logging
from typing import Literal
import argparse
import json
from dataclasses import replace

from map_utils import eval_map_points
from osc_messenger import OSCMessenger
from point import Point
from decisionTree import step_simple_decision_tree, step_advanced_decision_tree
from utility_ai import utility_agent, DEFAULT_PARAMS
from training import UtilityTrainer, PARAMS_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser()

ai_choices = [
    "simple_tree",
    "advanced_tree",
    "utility_function",
    "trained_utility_function",
    "train",
    "st",
    "at",
    "uf",
    "tuf",
]
ai_aliases = {
    "st": "simple_tree",
    "at": "advanced_tree",
    "uf": "utility_function",
    "tuf": "trained_utility_function",
}


def _normalize_ai(value: str) -> str:
    return ai_aliases.get(value, value)


parser.add_argument(
    "-p1",
    "--player1",
    type=_normalize_ai,
    choices=ai_choices,
    default="simple_tree",
)
parser.add_argument(
    "-p2",
    "--player2",
    type=_normalize_ai,
    choices=ai_choices,
    default="utility_function",
)
parser.add_argument(
    "-n",
    "--iterations",
    type=int,
    default=50,
    help="Number of training matches (default: 50)",
)

args = parser.parse_args()


ai_modes = {"p1": args.player1, "p2": args.player2}

# Validate iterations
if args.iterations < 1:
    parser.error("--iterations must be greater than 0")

# Only one bot can be trained at the same time
if list(ai_modes.values()).count("train") > 1:
    parser.error("Only one player can use 'train' mode.")

# Get id of trained player if any
train_player_id = None

if args.player1 == "train":
    train_player_id = 0
elif args.player2 == "train":
    train_player_id = 1

# Utility function trainer if needed
trainer = (
    UtilityTrainer(
        mode="train",
        player_id=train_player_id,
        max_trials=args.iterations,
    )
    if train_player_id is not None
    else None
)

# Load train parameters if needed
trained_params = None

if "trained_utility_function" in ai_modes.values():
    with PARAMS_FILE.open("r", encoding="utf-8") as f:
        saved_params = json.load(f)

    trained_params = replace(DEFAULT_PARAMS, **saved_params)


# Flag used for preventing multiple trainer updates caused by multiple end packets
match_active: bool = False

# Flag used for choosing correct action after last end response was lost
last_end_response: Literal["ok", "reset"] | None = None


def run_ai(
    mode: str,
    my_pos: Point,
    enemy_pos: Point,
    my_flag: bool,
    enemy_flag: bool,
    my_map,
) -> Point:

    if mode == "simple_tree":
        return step_simple_decision_tree(
            my_pos,
            enemy_pos,
            my_flag,
            enemy_flag,
            my_map,
        )

    elif mode == "advanced_tree":
        return step_advanced_decision_tree(
            my_pos,
            enemy_pos,
            my_flag,
            enemy_flag,
            my_map,
        )

    elif mode == "utility_function":
        return utility_agent(
            my_pos,
            enemy_pos,
            my_flag,
            enemy_flag,
            my_map,
            DEFAULT_PARAMS,
        )

    elif mode == "trained_utility_function":
        if trained_params is None:
            raise RuntimeError("Trained parameters not loaded")

        return utility_agent(
            my_pos,
            enemy_pos,
            my_flag,
            enemy_flag,
            my_map,
            trained_params,
        )

    elif mode == "train":
        if trainer is None:
            raise RuntimeError("Trainer not initialized")

        return utility_agent(
            my_pos,
            enemy_pos,
            my_flag,
            enemy_flag,
            my_map,
            trainer.params,
        )

    else:
        raise ValueError(f"Unknown AI mode: {mode}")


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

    def on_init(
        packet_id: int, b1_x: float, b1_y: float, b2_x: float, b2_y: float
    ) -> None:
        """Calculates relative map points and replies 'ok'."""
        global match_active, last_end_response

        b1: Point = {"x": float(b1_x), "y": float(b1_y)}
        b2: Point = {"x": float(b2_x), "y": float(b2_y)}
        maps["p1"], maps["p2"] = eval_map_points(b1, b2)
        logger.info("Initialized match (packet_id=%d): B1=%s, B2=%s", packet_id, b1, b2)

        match_active = True
        last_end_response = None
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
        """Calls selected AI functions and sends destinations back to UE5."""
        if "p1" not in maps or "p2" not in maps:
            logger.warning(
                "Ignored 'step' (packet_id=%d): match not initialized yet.", packet_id
            )
            return

        p1_pos: Point = {"x": float(p1_x), "y": float(p1_y)}
        p2_pos: Point = {"x": float(p2_x), "y": float(p2_y)}
        p1_has_flag = bool(p1_flag)
        p2_has_flag = bool(p2_flag)

        p1_dest = run_ai(
            ai_modes["p1"],
            p1_pos,
            p2_pos,
            p1_has_flag,
            p2_has_flag,
            maps["p1"],
        )

        # P2 - wybrany algorytm
        p2_dest = run_ai(
            ai_modes["p2"],
            p2_pos,
            p1_pos,
            p2_has_flag,
            p1_has_flag,
            maps["p2"],
        )

        logger.debug(
            "Step %d, P1 mode: %s | P1 source: %s ----> P1 dest: %s",
            packet_id,
            ai_modes["p1"],
            p1_pos,
            p1_dest,
        )
        logger.debug(
            "         P2 mode: %s | P2 source: %s ----> P2 dest: %s \n",
            ai_modes["p2"],
            p2_pos,
            p2_dest,
        )
        osc.send_step(
            packet_id,
            p1_dest["x"],
            p1_dest["y"],
            p2_dest["x"],
            p2_dest["y"],
        )

    def on_end(packet_id: int, winner_id: int) -> None:
        """Acknowledges match end and resets stored maps."""
        global match_active, last_end_response

        logger.info(
            "Match ended (packet_id=%d). Winner Player ID: %d", packet_id, winner_id
        )

        if not match_active:
            if last_end_response == "ok":
                osc.send_ok(packet_id)
            elif last_end_response == "reset":
                osc.send_reset(packet_id)
            return

        maps.clear()

        if trainer is not None and trainer.finish_match(winner_id):
            last_end_response = "reset"
        else:
            last_end_response = "ok"

        match_active = False

        if last_end_response == "reset":
            osc.send_reset(packet_id)
        else:
            osc.send_ok(packet_id)

    osc.register_handler("init", on_init)
    osc.register_handler("step", on_step)
    osc.register_handler("end", on_end)

    osc.start()


if __name__ == "__main__":
    run_game_loop()
