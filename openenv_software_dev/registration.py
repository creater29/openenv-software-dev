"""
Gymnasium environment registration for openenv-software-dev.
Call register_envs() once at import time (done in __init__.py).
"""
import gymnasium as gym


def register_envs():
    """Register all environment variants with Gymnasium."""
    # Easy variant — short horizon, easy tasks only
    gym.register(
        id="SoftwareDev-easy-v0",
        entry_point="openenv_software_dev.env:SoftwareDevEnv",
        kwargs={"difficulty": "easy", "max_steps": 15},
        max_episode_steps=15,
    )

    # Default medium variant
    gym.register(
        id="SoftwareDev-v0",
        entry_point="openenv_software_dev.env:SoftwareDevEnv",
        kwargs={"difficulty": "medium", "max_steps": 30},
        max_episode_steps=30,
    )

    # Hard variant — all tasks, more steps, LLM grading enabled
    gym.register(
        id="SoftwareDev-hard-v0",
        entry_point="openenv_software_dev.env:SoftwareDevEnv",
        kwargs={"difficulty": "hard", "max_steps": 50},
        max_episode_steps=50,
    )
