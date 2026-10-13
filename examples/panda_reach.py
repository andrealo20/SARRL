"""Run one nominal seven-joint Panda trajectory and print physical diagnostics."""

import json

from sarrl.evaluation.panda_nominal import run_episode


def main():
    # This seed belongs to the existing engineering-test block.
    record, _ = run_episode(9804100)
    result = {
        "scope": "nominal simulator example, no adaptation or safety certificate",
        "seed": record["seed"],
        "control_steps": record["steps"],
        "success": record["success"],
        "final_joint_error_rad": record["joint_error_max_rad"],
        "final_tip_error_m": record["tip_error_m"],
        "unsafe_episode": record["unsafe_episode"],
        "controller_saturated_steps": record["controller_saturated_steps"],
    }
    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
