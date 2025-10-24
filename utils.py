# utils.py
"""
Utilidades generales del proyecto RL Super Mario Bros.
"""

import datetime
import numpy as np


def timestamp() -> str:
    """Devuelve string tipo '2025-10-16_21-45' para nombres de carpetas/modelos."""
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
