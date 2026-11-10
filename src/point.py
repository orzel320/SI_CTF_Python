"""Defines shared types for representing 2D coordinates."""

from typing import TypedDict


class Point(TypedDict):
    """Represent a 2D point with x and y coordinates."""

    x: float
    y: float
