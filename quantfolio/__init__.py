"""quantfolio: vectorized backtesting and multi-strategy portfolio research."""

from .backtest import BacktestResult, backtest
from .metrics import performance_summary
from .portfolio import combine_inverse_vol

__all__ = ["BacktestResult", "backtest", "performance_summary", "combine_inverse_vol"]
__version__ = "0.1.0"
