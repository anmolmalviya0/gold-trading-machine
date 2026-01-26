"""
TERMINAL - IRONCLAD Strategy Optimizer V2
===========================================
CRITICAL FIX: Model outputs max confidence ~0.46
Previous thresholds (0.60-0.80) were TOO HIGH - no trades generated.
This version uses LOWER thresholds (0.30-0.50) to match model output range.
"""
import itertools
import pandas as pd
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest_engine import BacktestEngine

def run_ironclad_optimization():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          TERMINAL - NEGATIVE ALPHA TEST                 ║
    ║        INVERTED LOGIC: BUY Signal → SHORT Trade          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # 1. DEFINE THE REVISED GRID
    # Model outputs confidence ~0.46 max, so we use LOWER thresholds
    confidence_levels = [0.30, 0.35, 0.40, 0.45, 0.50]  # LOWERED to match model
    stop_loss_atrs = [1.5, 2.0]
    take_profit_atrs = [2.0, 3.0]
    
    combinations = list(itertools.product(confidence_levels, stop_loss_atrs, take_profit_atrs))
    results = []
    
    print(f"🎯 Testing {len(combinations)} combinations (conf 0.30-0.50 range)")
    print("=" * 60)

    # 2. EXECUTION LOOP
    for i, (conf, sl, tp) in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] Conf={conf}, SL={sl}ATR, TP={tp}ATR... ", end="", flush=True)
        
        try:
            # Initialize Engine with custom model path
            engine = BacktestEngine(
                model_path='/Users/anmol/Desktop/gold/terminal_alpha/models/terminal_lstm.pth',
                initial_capital=10000,
                risk_per_trade=0.02
            )
            
            # Load data
            data_path = '/Users/anmol/Desktop/gold/market_data/PAXGUSDT_5m.csv'
            df = pd.read_csv(data_path)
            
            # Temporarily modify the threshold in the run method
            # We'll monkey-patch the predictor's threshold
            original_run = engine.run
            
            def patched_run(df, seq_len=60):
                """Patched run with custom threshold"""
                # Prepare data
                df_prep = engine.prepare_features(df)
                
                feature_cols = [
                    'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
                    'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50',
                    'rsi_norm', 'macd_hist', 'bb_position', 'atr_ratio', 'volume_ratio'
                ]
                while len(feature_cols) < 20:
                    feature_cols.append(feature_cols[-1])
                feature_cols = feature_cols[:20]
                
                import numpy as np
                features = df_prep[feature_cols].values.astype(np.float32)
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Sample for speed
                max_samples = 10000
                if len(df_prep) > max_samples:
                    df_prep = df_prep.tail(max_samples).reset_index(drop=True)
                    features = features[-max_samples:]
                
                # Initialize tracking
                capital = engine.initial_capital
                equity_curve = [capital]
                trades = []
                
                in_position = False
                entry_price = 0
                entry_idx = 0
                max_hold = 20
                
                for j in range(seq_len, len(df_prep) - max_hold):
                    window = features[j-seq_len:j]
                    prediction = engine.predictor.predict(window, threshold=conf)  # USE CUSTOM CONF
                    
                    # INVERTED LOGIC: Model predicts BUY → We go SHORT
                    if not in_position and prediction['approved']:
                        in_position = True
                        entry_price = df_prep['close'].iloc[j] * (1 - engine.slippage)  # SELL entry
                        entry_idx = j
                    
                    elif in_position:
                        current_price = df_prep['close'].iloc[j]
                        bars_held = j - entry_idx
                        
                        atr = df_prep['atr'].iloc[j]
                        # SHORT LOGIC: Stop loss ABOVE entry, Take profit BELOW
                        stop_loss = entry_price + sl * atr  # INVERTED
                        take_profit = entry_price - tp * atr  # INVERTED
                        
                        should_exit = (
                            bars_held >= max_hold or
                            current_price >= stop_loss or  # INVERTED: price rises = stop
                            current_price <= take_profit   # INVERTED: price falls = profit
                        )
                        
                        if should_exit:
                            exit_price = current_price * (1 + engine.slippage)  # BUY to cover
                            
                            position_size = capital * engine.risk_per_trade / (sl * atr) if atr > 0 else 0
                            # SHORT P&L: profit when price drops
                            pnl = (entry_price - exit_price) * position_size  # INVERTED
                            fees = (entry_price + exit_price) * position_size * engine.fee_rate
                            net_pnl = pnl - fees
                            
                            trades.append({'pnl': net_pnl, 'win': net_pnl > 0})
                            
                            capital += net_pnl
                            equity_curve.append(capital)
                            in_position = False
                            
                            if capital <= 0:
                                break
                
                # Calculate metrics
                if not trades:
                    return {
                        'total_trades': 0,
                        'win_rate': 0,
                        'profit_factor': 0,
                        'net_return': 0,
                        'max_drawdown': 0
                    }
                
                wins = sum(1 for t in trades if t['win'])
                win_rate = wins / len(trades) * 100
                
                gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
                gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
                
                net_return = (equity_curve[-1] - engine.initial_capital) / engine.initial_capital * 100
                
                peak = engine.initial_capital
                max_dd = 0
                for eq in equity_curve:
                    if eq > peak:
                        peak = eq
                    dd = (peak - eq) / peak * 100
                    if dd > max_dd:
                        max_dd = dd
                
                return {
                    'total_trades': len(trades),
                    'wins': wins,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'net_return': net_return,
                    'max_drawdown': max_dd
                }
            
            # Load model
            if not engine.load_model():
                raise Exception("Model load failed")
            
            # Run patched backtest
            metrics = patched_run(df)
            
            # Extract results
            trades_count = metrics.get('total_trades', 0)
            win_rate = metrics.get('win_rate', 0.0)
            profit_factor = metrics.get('profit_factor', 0.0)
            net_return = metrics.get('net_return', 0.0)
            
            results.append({
                'confidence': conf,
                'sl_atr': sl,
                'tp_atr': tp,
                'trades': trades_count,
                'win_rate': round(win_rate, 1),
                'profit_factor': round(profit_factor, 2),
                'net_return': round(net_return, 1)
            })
            
            # Print immediate feedback with color coding
            if profit_factor > 1.5:
                print(f"🏆 Trades:{trades_count}, WR:{win_rate:.1f}%, PF:{profit_factor:.2f}, Ret:{net_return:+.1f}%")
            elif profit_factor > 1.0:
                print(f"✅ Trades:{trades_count}, WR:{win_rate:.1f}%, PF:{profit_factor:.2f}, Ret:{net_return:+.1f}%")
            elif trades_count == 0:
                print(f"⚪ No trades")
            else:
                print(f"❌ Trades:{trades_count}, WR:{win_rate:.1f}%, PF:{profit_factor:.2f}, Ret:{net_return:+.1f}%")

        except Exception as e:
            print(f"⚠️ CRASH: {str(e)[:50]}")
            results.append({
                'confidence': conf,
                'sl_atr': sl,
                'tp_atr': tp,
                'trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'net_return': 0.0,
                'error': str(e)
            })
        
        # AUTO-SAVE after each iteration
        if results:
            pd.DataFrame(results).to_csv('/Users/anmol/Desktop/gold/terminal_alpha/logs/negative_alpha_results.csv', index=False)

    # 3. FINAL REPORT
    print("\n" + "=" * 60)
    print("🏆 IRONCLAD OPTIMIZATION COMPLETE")
    print("=" * 60)
    
    if results:
        df_results = pd.DataFrame(results)
        print(f"\n📁 Results saved to: ironclad_v2_results.csv")
        
        # Sort and display
        df_results = df_results.sort_values(by='profit_factor', ascending=False)
        print("\n📊 ALL RESULTS (Ranked by Profit Factor):")
        print("-" * 60)
        print(df_results.to_string(index=False))
        
        # Find Champion
        valid = df_results[df_results['trades'] > 5]
        if not valid.empty:
            best = valid.iloc[0]
            print("\n" + "=" * 60)
            print("🏆 CHAMPION PARAMETER SET:")
            print("=" * 60)
            print(f"   Confidence: {best['confidence']}")
            print(f"   Stop Loss: {best['sl_atr']} ATR")
            print(f"   Take Profit: {best['tp_atr']} ATR")
            print(f"\n   Trades: {best['trades']}")
            print(f"   Win Rate: {best['win_rate']}%")
            print(f"   Profit Factor: {best['profit_factor']}")
            print(f"   Net Return: {best['net_return']}%")
            
            if best['profit_factor'] > 1.5:
                print("\n   🚀 VERDICT: DEPLOY READY (PF > 1.5)")
            elif best['profit_factor'] > 1.0:
                print("\n   ⚠️ VERDICT: EDGE EXISTS BUT WEAK (PF 1.0-1.5)")
            else:
                print("\n   ❌ VERDICT: NO EDGE - DO NOT DEPLOY")
        else:
            print("\n💀 NO PROFITABLE SETTINGS FOUND (with >5 trades).")
    else:
        print("\n💀 CRITICAL FAILURE: No results generated.")
    
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_ironclad_optimization()
    except KeyboardInterrupt:
        print("\n🛑 Optimization stopped by user.")
