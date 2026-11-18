import json
from dataclasses import asdict, replace
from pathlib import Path

import optuna

from utility_ai import DEFAULT_PARAMS


PARAMS_FILE = Path(__file__).with_name("best_utility_params.json")

TRAINABLE_PARAMS = {
    "w_travel": (0.05, 1.5),
    "w_threat": (0.1, 8.0),
    "w_int": (0.5, 6.0),
    "w_terr": (0.0, 1.5),
}


class UtilityTrainer:
    def __init__(
        self,
        mode: str = "train",
        player_id: int = 0,
        max_trials: int = 50,
    ):
        if mode not in ("train", "trained"):
            raise ValueError(f"Unknown training mode: {mode}")

        if player_id not in (0, 1):
            raise ValueError("player_id must be 0 or 1")

        if max_trials < 1:
            raise ValueError("max_trials must be greater than 0")

        self.mode = mode
        self.player_id = player_id
        self.max_trials = max_trials

        self.study = None
        self.trial = None
        self.params = DEFAULT_PARAMS

        if mode == "trained":
            with PARAMS_FILE.open("r", encoding="utf-8") as f:
                saved_params = json.load(f)

            self.params = replace(DEFAULT_PARAMS, **saved_params)

            print("LOADED TRAINED PARAMETERS")
            print(self.params)

        elif mode == "train":
            self.study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=1, n_startup_trials=4),
            )

            self.study.enqueue_trial(
                {name: getattr(DEFAULT_PARAMS, name) for name in TRAINABLE_PARAMS}
            )

            print("TRAINING STARTED")
            print(f"Player ID: {self.player_id}")
            print(f"Number of trials: {self.max_trials}")

            self.start_next_trial()

    def start_next_trial(self):
        completed = self.study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))

        if len(completed) >= self.max_trials:
            self.trial = None

            self.params = replace(
                DEFAULT_PARAMS,
                **self.study.best_params,
            )

            with PARAMS_FILE.open("w", encoding="utf-8") as f:
                json.dump(asdict(self.params), f, indent=4)

            print("TRAINING COMPLETED")
            print("Best score:", self.study.best_value)
            print("Best params:", self.study.best_params)
            print("Saved to:", PARAMS_FILE)
            return

        self.trial = self.study.ask()

        suggested_params = {
            name: self.trial.suggest_float(name, low, high)
            for name, (low, high) in TRAINABLE_PARAMS.items()
        }

        self.params = replace(
            DEFAULT_PARAMS,
            **suggested_params,
        )

        print(f"Starting trial {self.trial.number}")
        print(self.params)

    def finish_match(self, winner_id: int) -> bool:

        if self.mode != "train" or self.trial is None:
            return False

        if winner_id not in (0, 1):
            raise ValueError(f"Invalid winner_id: {winner_id}")

        reward = self.reward_function(winner_id)

        print(f"Trial {self.trial.number}: reward = {reward}")

        self.study.tell(self.trial, reward)
        self.start_next_trial()

        return self.trial is not None

    def reward_function(self, winner_id: int) -> float:
        return 1.0 if winner_id == self.player_id else 0.0
