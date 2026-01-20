"""
Walk-Forward Validator - Detect overfitting
"""

import pandas as pd
import logging
from typing import Dict, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-forward validation splits data into train/val/test.
    Ensures results are NOT lucky on one year.
    """
    
    def __init__(self, config: Dict):
        self.cfg = config['walkforward']
    
    def create_splits(self, df: pd.DataFrame, symbol: str = 'PAXGUSDT') -> List[Dict]:
        """
        Create walk-forward splits from OHLCV dataframe.
        
        Expected splits in config:
        - '2021-2023': train
        - '2024': validation
        - '2025': test
        
        Returns: [{
            'name': str,
            'stage': 'train' | 'validation' | 'test',
            'df': pd.DataFrame,
            'date_range': (start, end)
        }]
        """
        splits = []
        
        for split_cfg in self.cfg.get('splits', []):
            name = split_cfg.get('name')
            
            if 'train_start' in split_cfg:
                # Training split
                start = split_cfg['train_start']
                end = split_cfg['train_end']
                stage = 'train'
            elif 'validation_start' in split_cfg:
                # Validation split
                start = split_cfg['validation_start']
                end = split_cfg['validation_end']
                stage = 'validation'
            elif 'test_start' in split_cfg:
                # Test split
                start = split_cfg['test_start']
                end = split_cfg['test_end']
                stage = 'test'
            else:
                continue
            
            # Filter DataFrame
            mask = (df['timestamp'] >= pd.Timestamp(start)) & \
                   (df['timestamp'] <= pd.Timestamp(end))
            split_df = df[mask].reset_index(drop=True)
            
            if len(split_df) == 0:
                logger.warning(f"Empty split: {name} ({start} to {end})")
                continue
            
            splits.append({
                'name': name,
                'stage': stage,
                'df': split_df,
                'date_range': (start, end),
                'num_candles': len(split_df)
            })
            
            logger.info(f"Split {name} ({stage}): {len(split_df)} candles")
        
        return splits
    
    def validate_results(self, results: Dict[str, Dict]) -> Dict:
        """
        Compare results across splits.
        
        Results structure: {
            '2021-2023': {'win_rate': 55.2, 'pf': 1.3, ...},
            '2024': {'win_rate': 52.1, 'pf': 1.2, ...},
            '2025': {'win_rate': 51.8, 'pf': 1.15, ...}
        }
        
        Returns validation report with overfitting analysis.
        """
        
        if not results:
            return {'status': 'NO_RESULTS', 'overfit_risk': 'HIGH'}
        
        # Extract win rates
        win_rates = {}
        pfs = {}
        
        for split, metrics in results.items():
            win_rates[split] = metrics.get('win_rate', 0)
            pfs[split] = metrics.get('profit_factor', 0)
        
        # Check for stability
        wr_values = list(win_rates.values())
        pf_values = list(pfs.values())
        
        if not wr_values:
            return {'status': 'INSUFFICIENT_DATA', 'overfit_risk': 'HIGH'}
        
        wr_std = pd.Series(wr_values).std() if len(wr_values) > 1 else 0
        wr_mean = pd.Series(wr_values).mean()
        
        # Stability check
        if wr_std > 10:  # >10% std = unstable
            overfit_risk = 'HIGH'
        elif wr_std > 5:
            overfit_risk = 'MEDIUM'
        else:
            overfit_risk = 'LOW'
        
        # Check degradation (train vs test)
        train_wr = win_rates.get('2021-2023', 0)
        test_wr = win_rates.get('2025', 0)
        
        degradation = train_wr - test_wr if train_wr > 0 else 0
        
        # Verdict
        if overfit_risk == 'LOW' and degradation < 5:
            verdict = "VALID"
        elif overfit_risk == 'MEDIUM' or degradation < 10:
            verdict = "CAUTIOUS"
        else:
            verdict = "OVERFIT"
        
        return {
            'status': 'ANALYZED',
            'verdict': verdict,
            'overfit_risk': overfit_risk,
            'win_rate_mean': round(wr_mean, 2),
            'win_rate_std': round(wr_std, 2),
            'win_rate_by_split': win_rates,
            'profit_factor_by_split': pfs,
            'train_test_degradation': round(degradation, 2),
            'details': {
                'stable': wr_std <= 5,
                'profitable_all': all(wr > 50 for wr in wr_values if wr > 0),
                'consistent_pf': pd.Series(pf_values).std() < 0.3 if pf_values else False
            }
        }
    
    def print_validation_report(self, validation_results: Dict):
        """Pretty-print validation results"""
        
        report = f"""
╔════════════════════════════════════════════════════════════════════╗
║ WALK-FORWARD VALIDATION REPORT                                     ║
╠════════════════════════════════════════════════════════════════════╣
║ Status:         {validation_results.get('status', 'UNKNOWN'):<50} ║
║ Verdict:        {validation_results.get('verdict', 'UNKNOWN'):<50} ║
║ Overfit Risk:   {validation_results.get('overfit_risk', 'UNKNOWN'):<50} ║
╠════════════════════════════════════════════════════════════════════╣
║ Win Rate Mean:  {validation_results.get('win_rate_mean', 0):<10.2f}%                          ║
║ Win Rate Std:   {validation_results.get('win_rate_std', 0):<10.2f}%                          ║
╠════════════════════════════════════════════════════════════════════╣
║ BY SPLIT:                                                          ║
"""
        
        for split, wr in validation_results.get('win_rate_by_split', {}).items():
            pf = validation_results.get('profit_factor_by_split', {}).get(split, 0)
            report += f"║   {split:<20}: WR={wr:>6.2f}%  PF={pf:>5.2f}             ║\n"
        
        report += f"""║ Train-Test Degradation: {validation_results.get('train_test_degradation', 0):<5.2f}%              ║
╠════════════════════════════════════════════════════════════════════╣
║ Verdict:                                                           ║
║ - If VALID: Results are reliable, not overfit.                     ║
║ - If CAUTIOUS: Monitor live performance, possible overfitting.     ║
║ - If OVERFIT: Model is likely lucky on training data.             ║
╚════════════════════════════════════════════════════════════════════╝
        """
        
        print(report)
