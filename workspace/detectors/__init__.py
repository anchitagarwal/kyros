"""Project Kyros — ICT detector library (Phase 1: Sensory Foundation).

Pure, stateless detectors. Public interface is list[dict] in, list[dict] out.
No broker, IBKR, DB, or order-execution code. pandas/numpy used internally
only (none required at the boundary).

Modules:
    candles           — validate_candles, candle_metrics
    market_structure  — detect_swings, detect_bos, detect_choch
    fair_value_gaps   — detect_fvg, detect_ifvg
    displacement      — detect_displacement
    sessions          — detect_sessions, detect_kill_zones
    volume_imbalance  — detect_volume_imbalance, detect_opening_gaps
    liquidity         — detect_equal_levels, detect_prior_levels,
                        detect_liquidity_sweeps
    premium_discount  — detect_premium_discount
    order_blocks      — detect_order_blocks, detect_breaker_blocks
    inducement        — detect_turtle_soup, detect_inducement
    power_of_three    — detect_power_of_three
"""

from .candles import validate_candles, candle_metrics
from .market_structure import detect_swings, detect_bos, detect_choch
from .fair_value_gaps import detect_fvg, detect_ifvg
from .displacement import detect_displacement
from .sessions import detect_sessions, detect_kill_zones
from .volume_imbalance import detect_volume_imbalance, detect_opening_gaps
from .liquidity import (
    detect_equal_levels,
    detect_prior_levels,
    detect_liquidity_sweeps,
)
from .premium_discount import detect_premium_discount
from .order_blocks import detect_order_blocks, detect_breaker_blocks
from .inducement import detect_turtle_soup, detect_inducement
from .power_of_three import detect_power_of_three

__all__ = [
    "validate_candles",
    "candle_metrics",
    "detect_swings",
    "detect_bos",
    "detect_choch",
    "detect_fvg",
    "detect_ifvg",
    "detect_displacement",
    "detect_sessions",
    "detect_kill_zones",
    "detect_volume_imbalance",
    "detect_opening_gaps",
    "detect_equal_levels",
    "detect_prior_levels",
    "detect_liquidity_sweeps",
    "detect_premium_discount",
    "detect_order_blocks",
    "detect_breaker_blocks",
    "detect_turtle_soup",
    "detect_inducement",
    "detect_power_of_three",
]
