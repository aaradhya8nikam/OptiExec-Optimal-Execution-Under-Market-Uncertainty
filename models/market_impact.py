"""
Market Impact Model (Almgren-Chriss 2000 framework).
Handles linear and power-law temporary impact, permanent impact, and half-spread friction.
"""

from typing import Tuple, Optional
import numpy as np


def estimate_almgren_impact_params(
    price: float,
    adv: float,
    daily_vol: float,
    trading_minutes_per_day: float = 375.0
) -> Tuple[float, float]:
    """
    Parametric estimation of temporary impact eta (η) and permanent impact gamma (γ).
    Based on empirical calibrations from Almgren, Thum, Hauptmann, and Li (2005):
      γ ~= 0.1 * (daily_vol * price) / ADV
      η ~= 0.5 * (daily_vol * price) / (0.1 * ADV)
    Units:
      γ: [Currency / Share] (per share permanent shift)
      η: [Currency / (Share / Min)] (temporary friction per unit trading rate)
    """
    safe_adv = max(adv, 1000.0)
    gamma = 0.1 * (daily_vol * price) / safe_adv
    # Convert eta to per minute rate
    daily_volume_rate = safe_adv / trading_minutes_per_day
    eta = 0.5 * (daily_vol * price) / (0.1 * daily_volume_rate)
    return float(eta), float(gamma)


class MarketImpactModel:
    """
    Computes temporary and permanent price impact for given trade sizes and trading rates.
    """
    def __init__(
        self,
        eta: float = 2.5e-7,
        gamma: float = 5.0e-8,
        half_spread_bps: float = 2.5,
        power_alpha: float = 1.0  # 1.0 = linear impact (Almgren-Chriss); 0.5 = square root law
    ):
        self.eta = eta
        self.gamma = gamma
        self.half_spread_bps = half_spread_bps
        self.power_alpha = power_alpha

    def temporary_impact(self, trade_rate: float, price: float) -> float:
        """
        Temporary price impact h(v) in absolute price per share:
        h(v) = half_spread + eta * sign(v) * |v|^alpha
        """
        half_spread = price * (self.half_spread_bps * 1e-4)
        if abs(trade_rate) < 1e-9:
            return 0.0
        
        sign = np.sign(trade_rate)
        mag = abs(trade_rate) ** self.power_alpha
        return float(sign * half_spread + self.eta * sign * mag)

    def permanent_impact(self, trade_size: float) -> float:
        """
        Permanent price impact g(n) per trade interval:
        g(n) = gamma * n
        """
        return float(self.gamma * trade_size)

    def expected_execution_price(
        self,
        current_market_price: float,
        trade_size: float,
        interval_min: float,
        side_sign: float = 1.0  # +1 for BUY, -1 for SELL
    ) -> float:
        """
        Calculates the effective execution price ~S_k for a given trade slice:
        For BUY (+1):  ~S_k = S_{k-1} + h(v_k)
        For SELL (-1): ~S_k = S_{k-1} - h(v_k)
        """
        trade_rate = abs(trade_size) / max(interval_min, 1e-4)
        h_cost = self.temporary_impact(trade_rate, current_market_price)
        return float(current_market_price + side_sign * h_cost)


def calculate_instantaneous_cost(
    trade_size: float,
    current_price: float,
    interval_min: float,
    impact_model: MarketImpactModel,
    side_sign: float = 1.0
) -> float:
    """
    Computes dollar transaction friction cost for executing `trade_size` shares.
    Cost = trade_size * [ half_spread + eta * (trade_size / tau)^alpha ]
    """
    trade_rate = trade_size / max(interval_min, 1e-4)
    h_impact = impact_model.temporary_impact(trade_rate, current_price)
    return float(trade_size * h_impact)
