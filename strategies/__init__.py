"""
Strategies module for Optimal Execution Engine.
Includes Base Strategy, Immediate, TWAP, VWAP, and Almgren-Chriss Optimal Execution.
"""

from .base import BaseExecutionStrategy
from .immediate import ImmediateExecutionStrategy
from .twap import TWAPExecutionStrategy
from .vwap import VWAPExecutionStrategy
from .almgren_chriss import AlmgrenChrissExecutionStrategy
