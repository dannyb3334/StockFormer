import torch
import numpy as np
import pandas as pd
import pickle
import argparse
import matplotlib.pyplot as plt
import yaml
from StockFormer import create_compiled_stockformer, output_to_raw_numpy, output_to_signals


class StockFormerBacktester:
    def __init__(self, model_path, data_path, period_idx=-1, lookahead=0, model_params=None):
        """
        Initialize the backtester with model and data paths, period index, lookahead, and model parameters.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.num_stocks = None
        self.num_features = None
        self.seq_len = None
        self.pred_len = None
        self.hidden_dim = None
        self.model_path = model_path
        self.period_idx = period_idx
        self.lookahead = lookahead

        self._load_data(data_path, model_params)

    def _load_data(self, data_file='period_splits.pkl', model_params=None):
        """
        Load period split data, extract data dimensions, and initialize the model.

        Args:
            data_file: Path to the preprocessed data file containing period splits.
            model_params: Parameters for initializing the model.

        Returns:
            None. Sets self.period_splits and initializes self.model.
        """
        with open(data_file, 'rb') as f:
            self.period_splits = pickle.load(f)
        
        if not self.period_splits:
            raise ValueError("The file is empty or invalid.")
        
        first_period = next(iter(self.period_splits.values()))
        if 'training' not in first_period or 'X' not in first_period['training']:
            raise ValueError(f"Expected 'training' -> 'X' but got: {list(first_period.keys())}")
        
        sample_X = first_period['training']['X']
        sample_Y = first_period['training']['Y']
        
        if len(sample_X) == 0:
            raise ValueError("No training data found in the first period.")
        
        # Extract data dimensions
        self.seq_len = sample_X.shape[1]
        self.num_stocks = sample_X.shape[2]
        self.num_features = sample_X.shape[3]
        self.pred_len = sample_Y.shape[1]

        print(f"Data loaded successfully:")
        print(f"-- Periods: {len(self.period_splits)}")
        print(f"-- Sequence length: {self.seq_len}")
        print(f"-- Number of stocks: {self.num_stocks}")
        print(f"-- Number of features: {self.num_features}")
        print(f"-- Prediction horizon: {self.pred_len}")
        
        # Initialize model with provided parameters
        self.model = create_compiled_stockformer(**model_params)

        return self.period_splits

    def get_signals(self, X_test, Ts_test):
        """
        Generate trading signals and predictions from model outputs.

        Args:
            X_test: Test input features.
            Ts_test: Test time series features.

        Returns:
            signals: Trading signals (-1: sell, 0: hold, 1: buy)
            reg_pred: Regression predictions (return rates)
            cla_pred: Classification predictions (trend direction)
            reg_l_pred: Long-term regression predictions
            cla_l_pred: Long-term classification predictions
        """
        self.model.eval()

        with torch.inference_mode():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            Ts_tensor = torch.FloatTensor(Ts_test).to(self.device)
            out = self.model(X_tensor, Ts_tensor)


        # Debugging: print information about the outputs
        reg_pred, reg_l_pred, cla_pred, cla_l_pred, cla_probs, cla_l_probs = output_to_raw_numpy(out, self.lookahead)

        # Check for NaN values in predictions
        if np.any(np.isnan(reg_pred)) or np.any(np.isnan(cla_pred)):
            print("Warning: NaN values detected in predictions")
            return np.zeros_like(reg_pred), reg_pred, cla_pred
        
        # Analyze unique values in classification predictions
        cla_unique = np.unique(cla_pred)
        cla_l_unique = np.unique(cla_l_pred)
        
        print(f"Classification prediction analysis:")
        print(f"-- cla_pred unique values: {cla_unique} (count: {len(cla_unique)})")
        print(f"-- cla_l_pred unique values: {cla_l_unique} (count: {len(cla_l_unique)})")
        
        if len(cla_unique) == 1:
            print(f"Warning: All short-term classification predictions are identical (value: {cla_unique[0]})")
        if len(cla_l_unique) == 1:
            print(f"Warning: All long-term classification predictions are identical (value: {cla_l_unique[0]})")

        # Generate trading signals based on classification probabilities and regression predictions
        signals = output_to_signals(out, self.lookahead)

        total_signals = len(reg_pred) * self.num_stocks
        buy_count = np.sum(cla_pred == 1)
        sell_count = np.sum(cla_pred == 0)
        
        print(f"Summary:")
        print(f"-- Buy signals: {buy_count}/{total_signals} ({100*buy_count/total_signals:.1f}%)")
        print(f"-- Sell signals: {sell_count}/{total_signals} ({100*sell_count/total_signals:.1f}%)")
        
        return signals, reg_pred, cla_pred, reg_l_pred, cla_l_pred

    def calculate_baseline_returns(self, reg_actual):
        """
        Calculate cumulative and daily returns for an equal-weighted (hold-all) portfolio.

        Args:
            reg_actual: Actual returns (time_steps, num_stocks)

        Returns:
            baseline_returns: Cumulative returns for the baseline strategy
            baseline_daily_returns: Daily returns for the baseline strategy
        """
        baseline_daily_returns = np.mean(reg_actual, axis=1)
        baseline_returns = np.cumprod(1 + baseline_daily_returns) - 1
        return baseline_returns, baseline_daily_returns

    def calculate_strategy_returns(self, signals, reg_actual):
        """
        Calculate cumulative and daily returns for the strategy based on trading signals.

        Args:
            signals: Trading signals (time_steps, num_stocks)
            reg_actual: Actual returns (time_steps, num_stocks)

        Returns:
            strategy_returns: Cumulative returns for the strategy
            strategy_daily_returns: Daily returns for the strategy
            holdings: Position holdings over time (binary matrix)
        """
        time_steps, num_stocks = signals.shape
        holdings = np.zeros((time_steps, num_stocks))
        strategy_daily_returns = np.zeros(time_steps)

        for t in range(time_steps):
            held_stocks = (signals[t] == 1)
            holdings[t] = held_stocks.astype(float)
            if np.any(held_stocks):
                strategy_daily_returns[t] = np.mean(reg_actual[t, held_stocks])
            else:
                strategy_daily_returns[t] = 0.0

        strategy_returns = np.cumprod(1 + strategy_daily_returns) - 1
        return strategy_returns, strategy_daily_returns, holdings

    def backtest_period(self, period_data, period_idx):
        """
        Perform backtesting and evaluation for a single period.

        Args:
            period_data: Data for the selected period.
            period_idx: Index of the period.

        Returns:
            Dictionary with portfolio performance metrics and analysis.
        """
        X_test = period_data['test']['X']
        Y_test = period_data['test']['Y']
        Ts_test = period_data['test']['Ts']
        
        if len(X_test) == 0 or len(Y_test) == 0 or len(Ts_test) == 0:
            print(f"Invalid test data available for period {period_idx}")
            return None
        
        signals, reg_pred, cla_pred, reg_l_pred, cla_l_pred = self.get_signals(X_test, Ts_test)
        reg_actual = Y_test[:, self.lookahead, :, 0]
        total_samples = reg_actual.size

        # Check for NaN values in predictions and actuals
        if np.any(np.isnan(reg_actual)):
            print("Warning: NaN values detected in reg_actual")
        if np.any(np.isnan(reg_pred)):
            print("Warning: NaN values detected in reg_pred")
        if np.any(np.isnan(cla_pred)):
            print("Warning: NaN values detected in cla_pred")
        if np.any(np.isnan(reg_l_pred)):
            print("Warning: NaN values detected in reg_l_pred")
        if np.any(np.isnan(cla_l_pred)):
            print("Warning: NaN values detected in cla_l_pred")
        
        print("=== BACKTEST RESULTS ===")
        # Evaluate prediction accuracy
        matching_signs = np.sum((reg_actual > 0) == (cla_pred > 0))
        print(f"Number of times cla_pred matches the sign of reg_actual: {matching_signs}/{total_samples} ({100 * matching_signs / total_samples:.2f}%)")

        matching_signs = np.sum((reg_actual > 0) == (reg_pred > 0))
        print(f"Number of times reg_pred matches the sign of reg_actual: {matching_signs}/{total_samples} ({100 * matching_signs / total_samples:.2f}%)")

        matching_signs_l = np.sum((reg_actual > 0) == (cla_l_pred > 0))
        print(f"Number of times cla_l_pred matches the sign of reg_actual: {matching_signs_l}/{total_samples} ({100 * matching_signs_l / total_samples:.2f}%)")

        matching_signs_l = np.sum((reg_actual > 0) == (reg_l_pred > 0))
        print(f"Number of times reg_l_pred matches the sign of reg_actual: {matching_signs_l}/{total_samples} ({100 * matching_signs_l / total_samples:.2f}%)")

        avg_delta = np.mean(np.abs(reg_pred - reg_actual))
        print(f"Average delta between reg_pred and reg_actual: {avg_delta:.4f}")

        # Analyze variances for predictions and actuals
        cla_pred_flat = cla_pred.flatten()
        cla_l_pred_flat = cla_l_pred.flatten()
        reg_pred_flat = reg_pred.flatten()
        reg_l_pred_flat = reg_l_pred.flatten()
        reg_actual_flat = reg_actual.flatten()

        cla_pred_var = np.var(cla_pred_flat)
        cla_l_pred_var = np.var(cla_l_pred_flat)
        reg_pred_var = np.var(reg_pred_flat)
        reg_l_pred_var = np.var(reg_l_pred_flat)
        reg_actual_var = np.var(reg_actual_flat)
        
        print(f"Variances - cla_pred: {cla_pred_var:.6f}, cla_l_pred: {cla_l_pred_var:.6f}, reg_pred: {reg_pred_var:.6f}, reg_l_pred: {reg_l_pred_var:.6f}, reg_actual: {reg_actual_var:.6f}")
        
        # Calculate returns for baseline and strategy
        baseline_returns, baseline_daily_returns = self.calculate_baseline_returns(reg_actual)
        strategy_returns, strategy_daily_returns, holdings = self.calculate_strategy_returns(signals, reg_actual)
        
        print(f"\n=== PERFORMANCE COMPARISON ===")
        print(f"Baseline Strategy (Hold All Stocks):")
        print(f"  Total Return: {baseline_returns[-1]:.4f} ({baseline_returns[-1]*100:.2f}%)")
        print(f"  Daily Volatility: {np.std(baseline_daily_returns):.4f}")
        print(f"  Sharpe Ratio: {np.mean(baseline_daily_returns)/np.std(baseline_daily_returns):.4f}")
        
        print(f"\nStrategy:")
        print(f"  Total Return: {strategy_returns[-1]:.4f} ({strategy_returns[-1]*100:.2f}%)")
        print(f"  Daily Volatility: {np.std(strategy_daily_returns):.4f}")
        print(f"  Sharpe Ratio: {np.mean(strategy_daily_returns)/np.std(strategy_daily_returns):.4f}")
        
        print(f"\nStrategy Analysis:")
        avg_holdings = np.mean(np.sum(holdings, axis=1))
        utilization = avg_holdings / self.num_stocks * 100 if self.num_stocks else 0.0
        print(f"  Average number of stocks held: {avg_holdings:.2f}/{self.num_stocks}")
        print(f"  Portfolio utilization: {utilization:.1f}%")
        
        return {
            'period_idx': period_idx + 1,
            'baseline_returns': baseline_returns,
            'strategy_returns': strategy_returns,
            'baseline_daily_returns': baseline_daily_returns,
            'strategy_daily_returns': strategy_daily_returns,
            'holdings': holdings
        }

    def run_backtest(self):
        """
        Run the backtesting pipeline for the selected period, print summary, and plot results.
        """
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
       
        print("Starting backtest...")
        results = []
        
        period_idx, period_data = list(self.period_splits.items())[self.period_idx]
        period_result = self.backtest_period(period_data, period_idx)
        if period_result is not None:
            results.append(period_result)
        
        if results:
            self.generate_performance_summary(results)
            
        return results
        
    def generate_performance_summary(self, results):
        """
        Print summary statistics and plot performance for a single period.

        Args:
            results: List of period results from backtest_period (should be length 1)
        """
        if not results or len(results) != 1:
            print("Error: Only one period should be provided for summary.")
            return

        result = results[0]
        baseline_returns = result['baseline_returns']
        strategy_returns = result['strategy_returns']
        baseline_daily = result['baseline_daily_returns']
        strategy_daily = result['strategy_daily_returns']
        period_idx = result['period_idx']

        baseline_total_return = baseline_returns[-1]
        strategy_total_return = strategy_returns[-1]
        baseline_volatility = np.std(baseline_daily)
        strategy_volatility = np.std(strategy_daily)

        print(f"Period: {period_idx}")
        print(f"  Baseline (Hold All): {baseline_total_return:.4f} ({baseline_total_return*100:.2f}%)")
        print(f"  Strategy: {strategy_total_return:.4f} ({strategy_total_return*100:.2f}%)")
        print(f"  Outperformance: {strategy_total_return - baseline_total_return:.4f} ({(strategy_total_return - baseline_total_return)*100:.2f}%)")

        print(f"\nRisk Metrics:")
        print(f"  Baseline Volatility: {baseline_volatility:.4f}")
        print(f"  Strategy Volatility: {strategy_volatility:.4f}")

        if baseline_volatility > 0:
            baseline_sharpe = np.mean(baseline_daily) / baseline_volatility
            print(f"  Baseline Sharpe: {baseline_sharpe:.4f}")
        if strategy_volatility > 0:
            strategy_sharpe = np.mean(strategy_daily) / strategy_volatility
            print(f"  Strategy Sharpe: {strategy_sharpe:.4f}")

        self.plot_performance_comparison(result)
        
    def plot_performance_comparison(self, result):
        """
        Plot and save cumulative returns for baseline and strategy for a single period.

        Args:
            result: Dictionary with period results
        """
        baseline_returns = result['baseline_returns']
        strategy_returns = result['strategy_returns']
        period_idx = result['period_idx']
        time_steps = range(len(baseline_returns))

        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, baseline_returns, label='Baseline (Hold All)', linewidth=2, alpha=0.8)
        plt.plot(time_steps, strategy_returns, label='Strategy', linewidth=2, alpha=0.8)
        plt.title(f'Cumulative Returns - Period {period_idx}', fontsize=16, fontweight='bold')
        plt.xlabel('Time Steps')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        final_baseline = baseline_returns[-1]
        final_strategy = strategy_returns[-1]
        plt.text(0.02, 0.98, f'Baseline: {final_baseline*100:.1f}%\nStrategy: {final_strategy*100:.1f}%',
                 transform=plt.gca().transAxes, verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig('backtest_period.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\nSingle period chart saved as 'backtest_period.png'")
        
def main():
    """
    Main function to run the backtester using configuration from a YAML file.
    Loads config, initializes backtester, and runs backtest.
    """
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_path = config.get('data', 'period_splits.pkl')
    model_path = config.get('model_path', 'stockformer_model.pth')
    model_params = config.get('model_params', {})
    
    backtester = StockFormerBacktester(
        model_path=model_path,
        data_path=data_path,
        period_idx=-1,
        lookahead=0,
        model_params=model_params,
    )
    
    result_values = backtester.run_backtest()

if __name__ == "__main__":
    main()
