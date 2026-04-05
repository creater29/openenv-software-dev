"""
OpenEnv Software Development Environment
A Gymnasium-compatible RL environment for software development tasks.
"""
from .env import SoftwareDevEnv
from .registration import register_envs

register_envs()

__version__ = "0.1.0"
__all__ = ["SoftwareDevEnv"]
