"""
TERMINAL I - QUANT UTILS
=========================
Quantitative finance utilities for position sizing, risk management, and order calculation.
"""

def calc_position_size(balance: float, risk_pct: float, entry: float, stop_loss: float) -> float:
    """
    Calculate position size based on risk percentage and stop loss distance.
    
    Formula: (Balance * Risk%) / |Entry - StopLoss|
    
    Args:
        balance: Total account balance
        risk_pct: Risk percentage (e.g., 0.03 for 3%)
        entry: Entry price
        stop_loss: Stop loss price
        
    Returns:
        Position size (in asset units)
    """
    if entry <= 0 or stop_loss <= 0:
        return 0.0
        
    risk_amount = balance * risk_pct
    stop_distance = abs(entry - stop_loss)
    
    if stop_distance == 0:
        return 0.0
        
    position_size = risk_amount / stop_distance
    return position_size

def calc_kelly_criterion(win_rate: float, win_loss_ratio: float) -> float:
    """
    Calculate Kelly Criterion for optimal position sizing.
    
    Args:
        win_rate: Probability of winning (0.0 to 1.0)
        win_loss_ratio: Ratio of Avg Win / Avg Loss
        
    Returns:
        Optimal fraction of bankroll to wager
    """
    if win_loss_ratio == 0:
        return 0.0
        
    kelly = list_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
    return max(0.0, kelly)
