import torch
import numpy as np
import pandas as pd
import pickle
import argparse
import matplotlib.pyplot as plt
from StockFormer import StockFormer

MODEL_PATH = f"stockformer_model.pth"

class StockFormerBacktester:
    def __init__(self, initial_capital=100000, cost=0.001):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Starting capital for backtesting
            cost: Transaction cost as a fraction (e.g., 0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.cost = cost
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model will be initialized when loading data
        self.model = None
        self.num_stocks = None
        self.num_features = None
        self.seq_len = None
        self.pred_len = None
        self.hidden_dim = 128  # Default hidden dimension for StockFormer
            
    def load_data(self, data_file='period_splits.pkl'):
        """Load and initialize model based on data dimensions.
        
        Args:
            data_file: Path to the preprocessed data file containing period splits.
        Returns:
            period_splits: Dictionary with training and test data for each period.
        """
        # Load period splits
        with open(data_file, 'rb') as f:
            self.period_splits = pickle.load(f)
        
        # Check if period splits is empty
        if not self.period_splits:
            raise ValueError(f"The file is empty or invalid.")
        
        first_period = next(iter(self.period_splits.values()))
        if 'training' not in first_period or 'X' not in first_period['training']:
            raise ValueError(f"Expected 'training' -> 'X' but got: {list(first_period.keys())}")
        
        sample_X = first_period['training']['X']
        sample_Y = first_period['training']['Y']
        
        if len(sample_X) == 0:
            raise ValueError(f"No training data found in the first period.")
        
        # Print data shape
        # Input Shape: (batch, lag, num_stocks, num_features)
        # Output Shape: (batch, pred_len, num_stocks, num_features)
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
        
        # Initialize model
        self.model = StockFormer(num_stocks=10).to(self.device)
        
        return self.period_splits

    
    def generate_signals(self, X_test, Ts_test):
        """
        Generate trading signals from model predictions.
        
        Returns:
            signals: Trading signals (-1: sell, 0: hold, 1: buy)
            reg_pred: Regression predictions (return rates)
            cla_pred: Classification predictions (trend direction)
        """
        self.model.eval()

        with torch.inference_mode():
            # Format input tensors
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            Ts_tensor = torch.FloatTensor(Ts_test).to(self.device)
            out = self.model(X_tensor, Ts_tensor)

            reg_pred = out["cla"][:, 1, :].cpu().numpy() # (time, num_stocks)
            cla_pred = out["reg"][:, 1, :].cpu().numpy() # (time, num_stocks)

            # Flatten predictions for thresholding
            reg_pred_flat = reg_pred.flatten()
            cla_pred_flat = cla_pred.flatten()

            # Set batch thresholds for buy/sell signals
            reg_buy_threshold = np.percentile(reg_pred_flat, 80)
            reg_sell_threshold = np.percentile(reg_pred_flat, 20)
            cla_buy_threshold = np.percentile(cla_pred_flat, 70)
            cla_sell_threshold = np.percentile(cla_pred_flat, 30)
            
            # Generate signals based on both regression and classification predictions
            signals = np.zeros_like(reg_pred)
            for t in range(len(reg_pred)):
                for stock in range(self.num_stocks):
                    return_pred = reg_pred[t, stock]
                    trend_pred = cla_pred[t, stock]
                    
                    # Buy signal: high return prediction AND positive trend
                    if return_pred > reg_buy_threshold and trend_pred > cla_buy_threshold:
                        signals[t, stock] = 1
                    # Sell signal: low return prediction AND negative trend  
                    elif return_pred < reg_sell_threshold and trend_pred < cla_sell_threshold:
                        signals[t, stock] = -1
                    # Neither: Hold
                    else:
                        signals[t, stock] = 0
            
            # Print Summary
            total_signals = len(reg_pred) * self.num_stocks
            buy_count = np.sum(signals == 1)
            sell_count = np.sum(signals == -1)
            hold_count = np.sum(signals == 0)
            
            print(f"Summary:")
            print(f"-- Buy signals: {buy_count}/{total_signals} ({100*buy_count/total_signals:.1f}%)")
            print(f"-- Sell signals: {sell_count}/{total_signals} ({100*sell_count/total_signals:.1f}%)")
            print(f"-- Hold signals: {hold_count}/{total_signals} ({100*hold_count/total_signals:.1f}%)")
            print(f"-- Return pred. range: [{reg_pred.min():.4f}, {reg_pred.max():.4f}]")
            print(f"-- Trend pred. range: [{cla_pred.min():.4f}, {cla_pred.max():.4f}]")
            print(f"-- Buy thresholds: return>{reg_buy_threshold:.4f}, trend>{cla_buy_threshold:.4f}")
            print(f"-- Sell thresholds: return<{reg_sell_threshold:.4f}, trend<{cla_sell_threshold:.4f}")
            
            return signals, reg_pred, cla_pred

    def backtest_period(self, period_data, period_idx):
        """
        Backtest on a single period.

        Returns:
            results: Dictionary with portfolio performance metrics
        """
        # TODO: Implement backtest test with actual returns and improved risk management
        # TODO: Implement actual start and end dates/prices for each period
        
        # Get test data for the period
        X_test = period_data['test']['X']
        Y_test = period_data['test']['Y']
        Ts_test = period_data['test']['Ts']
        
        # Check if test data is empty
        if len(X_test) == 0 or len(Y_test) == 0 or len(Ts_test) == 0:
            print(f"Invalid test data available for period {period_idx}")
            return None
        
        # Generate signals
        signals, reg_pred, cla_pred = self.generate_signals(X_test, Ts_test)
        standardized_returns = Y_test[:, 0, :, 0]  # (samples, num_stocks) - standardized return rates
        
        # Get standardization statistics for proper inverse transformation
        target_stats = period_data['target_standardization']
        
        # Inverse transform standardized returns back to original scale
        returns = np.zeros_like(standardized_returns)
        for stock_idx in range(self.num_stocks):
            if stock_idx in target_stats:
                # Get the mean and std for Y_RETURN_RATE (index 0) for this stock
                stock_stats = target_stats[stock_idx]
                return_mean = stock_stats['mean'].iloc[0]  # Y_RETURN_RATE mean
                return_std = stock_stats['std'].iloc[0]    # Y_RETURN_RATE std
                
                # Inverse transform: actual_return = standardized_return * std + mean
                returns[:, stock_idx] = standardized_returns[:, stock_idx] * return_std + return_mean
            else:
                # Fallback if stats not available (shouldn't happen)
                returns[:, stock_idx] = standardized_returns[:, stock_idx] * 0.02

        print(f"Signal range: {signals.min():.1f} to {signals.max():.1f}")
        print(f"Standardized return range: {standardized_returns.min():.4f} to {standardized_returns.max():.4f}")
        print(f"Actual return range: {returns.min():.4f} to {returns.max():.4f}")
        print(f"Buy signals: {np.sum(signals == 1)}, Sell signals: {np.sum(signals == -1)}")

        # Initialize portfolio and baseline
        portfolio_value = self.initial_capital
        baseline_value = self.initial_capital
        cash = self.initial_capital
        positions = np.zeros(self.num_stocks)  # Number of shares held
        
        # Initialize baseline with equal allocation - more realistic baseline
        initial_stock_price = 100.0  # Assume initial stock price of $100
        baseline_shares_per_stock = self.initial_capital / (self.num_stocks * initial_stock_price)
        baseline_positions = np.full(self.num_stocks, baseline_shares_per_stock)
        baseline_cash = 0.0
        
        # Track performance
        portfolio_values = [portfolio_value]
        baseline_values = [baseline_value]
        trade_log = []
        
        # Keep track of price movements
        stock_prices = np.full(self.num_stocks, initial_stock_price)
        
        for t in range(len(signals)):
            current_signals = signals[t]
            current_returns = returns[t]  # Use actual inverse-transformed returns
            
            # Update stock prices based on actual returns
            stock_prices = stock_prices * (1 + current_returns)
            
            # Update portfolio value
            position_values = positions * stock_prices
            portfolio_value = cash + np.sum(position_values)
            
            # Update baseline value - buy and hold strategy
            baseline_position_values = baseline_positions * stock_prices
            baseline_value = baseline_cash + np.sum(baseline_position_values)
            
            # Risk management strategy - Fixed
            # Limit exposure to 10% of portfolio per position, max 80% total exposure
            buy_signals_count = np.sum(current_signals == 1)
            max_position_size = portfolio_value * 0.1
            max_total_exposure = portfolio_value * 0.8
            
            # Current exposure
            current_exposure = np.sum(positions * stock_prices)
            
            # For each stock, execute a Trade action
            for stock_idx in range(self.num_stocks):
                signal = current_signals[stock_idx]
                stock_price = stock_prices[stock_idx]
                
                # Buy signal with risk management
                if signal == 1 and buy_signals_count > 0 and cash > 1000:
                    # Calculate position size with risk management
                    available_cash = min(cash, max_total_exposure - current_exposure)
                    if available_cash > 0:
                        # Allocate available cash equally among buy signals, but cap per position
                        allocation = min(available_cash / buy_signals_count, max_position_size)
                        shares_to_buy = allocation / (stock_price * (1 + self.cost))
                        cost = shares_to_buy * stock_price * (1 + self.cost)
                        
                        # Execute buy if within limits
                        if cost <= cash and cost >= 100 and shares_to_buy > 0.01:
                            positions[stock_idx] += shares_to_buy
                            cash -= cost
                            current_exposure += shares_to_buy * stock_price
                            trade_log.append({
                                'period': period_idx,
                                'time': t,
                                'stock': stock_idx,
                                'action': 'BUY',
                                'shares': shares_to_buy,
                                'price': stock_price,
                                'cost': cost,
                                'signal_strength': reg_pred[t, stock_idx] if reg_pred is not None else 0
                            })
                # Sell signal with risk management
                elif signal == -1 and positions[stock_idx] > 0.01:
                    # Sell all positions in this stock
                    proceeds = positions[stock_idx] * stock_price * (1 - self.cost)
                    cash += proceeds
                    current_exposure -= positions[stock_idx] * stock_price
                    trade_log.append({
                        'period': period_idx,
                        'time': t,
                        'stock': stock_idx,
                        'action': 'SELL',
                        'shares': positions[stock_idx],
                        'price': stock_price,
                        'proceeds': proceeds,
                        'signal_strength': reg_pred[t, stock_idx] if reg_pred is not None else 0
                    })
                    positions[stock_idx] = 0
            
            # Update portfolio and baseline values
            position_values = positions * stock_prices
            portfolio_value = cash + np.sum(position_values)
            portfolio_values.append(portfolio_value)
            baseline_values.append(baseline_value)
        
        # Calculate performance metrics with safety checks
        initial_portfolio = portfolio_values[0]
        final_portfolio = portfolio_values[-1]
        initial_baseline = baseline_values[0]
        final_baseline = baseline_values[-1]

        if initial_portfolio > 0:
            portfolio_return = (final_portfolio - initial_portfolio) / initial_portfolio
        else:
            portfolio_return = 0.0
            
        if initial_baseline > 0:
            baseline_return = (final_baseline - initial_baseline) / initial_baseline
        else:
            baseline_return = 0.0
        
        # Calculate Sharpe ratio with safety checks
        portfolio_returns = np.diff(portfolio_values) / (np.array(portfolio_values[:-1]) + 1e-8)
        baseline_returns = np.diff(baseline_values) / (np.array(baseline_values[:-1]) + 1e-8)
        portfolio_sharpe = 0.0
        baseline_sharpe = 0.0
        
        if len(portfolio_returns) > 1:
            port_std = np.std(portfolio_returns)
            if port_std > 1e-8:
                portfolio_sharpe = np.mean(portfolio_returns) / port_std * np.sqrt(252)
        
        if len(baseline_returns) > 1:
            base_std = np.std(baseline_returns)
            if base_std > 1e-8:
                baseline_sharpe = np.mean(baseline_returns) / base_std * np.sqrt(252)
        
        results = {
            'period': period_idx,
            'portfolio_values': portfolio_values,
            'baseline_values': baseline_values,
            'portfolio_return': portfolio_return,
            'baseline_return': baseline_return,
            'portfolio_sharpe': portfolio_sharpe,
            'baseline_sharpe': baseline_sharpe,
            'num_trades': len(trade_log),
            'trade_log': trade_log,
            'final_cash': cash,
            'final_positions': positions
        }
        
        return results

    def run_backtest(self, data_file='period_splits.pkl'):
        """
        Run the backtesting pipeline.
        
        Args:
            data_file: Path to the preprocessed data file
        """
        
        # Load data
        self.load_data(data_file)
                
        # Load existing model - handle both dictionary and direct state_dict formats
        checkpoint = torch.load(MODEL_PATH, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Handle direct state_dict save format
            self.model.load_state_dict(checkpoint)
       
        # Run backtesting on each period
        print("Starting backtest...")
        results = []
        
        for period_idx, period_data in self.period_splits.items():
            print(f"Backtesting period {period_idx}...")
            result = self.backtest_period(period_data, period_idx)
            if result is not None:
                results.append(result)
                print(f"-- Portfolio Return: {result['portfolio_return']:.2%}")
                print(f"-- Baseline Return:  {result['baseline_return']:.2%}")
                print(f"-- Alpha: {result['portfolio_return'] - result['baseline_return']:.2%}")
                print(f"-- Number of trades: {result['num_trades']}")
        
        return results

    def plot_results(self, results, save_path='backtest_results.png'):
        """
        Plot backtest results with improved visualization.
        Args:
            results: List of backtest result dictionaries
            save_path: Path to save the plot image
        """
        # Plot 1: Portfolio vs Baseline Value Over Time
        # Plot 2: Returns Comparison
        # Plot 3: Sharpe Ratios
        # Plot 4: Cumulative Performance
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('StockFormer Backtest Results', fontsize=16, fontweight='bold')
        
        # Portfolio vs Baseline Value Over Time
        ax1 = axes[0, 0]
        for i, result in enumerate(results):
            time_steps = range(len(result['portfolio_values']))
            ax1.plot(time_steps, result['portfolio_values'], 
                    label=f"Portfolio P{result['period']}", alpha=0.8, linewidth=2)
            ax1.plot(time_steps, result['baseline_values'], 
                    label=f"Baseline P{result['period']}", alpha=0.6, linestyle='--')
        
        ax1.set_title('Portfolio Value vs Baseline')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Returns Comparison
        ax2 = axes[0, 1]
        periods = [r['period'] for r in results]
        portfolio_returns = [r['portfolio_return'] * 100 for r in results]
        baseline_returns = [r['baseline_return'] * 100 for r in results]
        
        x = np.arange(len(periods))
        width = 0.35
        
        ax2.bar(x - width/2, portfolio_returns, width, label='Portfolio', alpha=0.8, color='blue')
        ax2.bar(x + width/2, baseline_returns, width, label='Baseline', alpha=0.8, color='orange')
        
        ax2.set_title('Returns by Period')
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Return (%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(periods)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Sharpe Ratios
        ax3 = axes[1, 0]
        portfolio_sharpe = [r['portfolio_sharpe'] for r in results]
        baseline_sharpe = [r['baseline_sharpe'] for r in results]
        
        ax3.bar(x - width/2, portfolio_sharpe, width, label='Portfolio', alpha=0.8, color='green')
        ax3.bar(x + width/2, baseline_sharpe, width, label='Baseline', alpha=0.8, color='red')
        
        ax3.set_title('Sharpe Ratio by Period')
        ax3.set_xlabel('Period')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.set_xticks(x)
        ax3.set_xticklabels(periods)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Cumulative Performance
        ax4 = axes[1, 1]
        
        # Combine all periods for cumulative view
        all_portfolio_values = []
        all_baseline_values = []
        
        for result in results:
            if len(all_portfolio_values) == 0:
                all_portfolio_values.extend(result['portfolio_values'])
                all_baseline_values.extend(result['baseline_values'])
            else:
                # Normalize to continue from previous period
                scale_p = all_portfolio_values[-1] / result['portfolio_values'][0]
                scale_b = all_baseline_values[-1] / result['baseline_values'][0]
                
                all_portfolio_values.extend([v * scale_p for v in result['portfolio_values'][1:]])
                all_baseline_values.extend([v * scale_b for v in result['baseline_values'][1:]])
        
        time_steps = range(len(all_portfolio_values))
        ax4.plot(time_steps, all_portfolio_values, label='Portfolio', linewidth=2, color='blue')
        ax4.plot(time_steps, all_baseline_values, label='Baseline (Buy & Hold)', linewidth=2, color='orange')
        
        ax4.set_title('Cumulative Performance')
        ax4.set_xlabel('Time Steps (All Periods)')
        ax4.set_ylabel('Portfolio Value ($)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results saved to {save_path}")
        plt.show()
        
    def print_summary(self, results):
        """
        Print comprehensive summary statistics.

        Args:
            results: List of backtest result dictionaries
        """
        
        total_portfolio_return = np.mean([r['portfolio_return'] for r in results])
        total_baseline_return = np.mean([r['baseline_return'] for r in results])
        alpha = total_portfolio_return - total_baseline_return
        
        avg_portfolio_sharpe = np.mean([r['portfolio_sharpe'] for r in results])
        avg_baseline_sharpe = np.mean([r['baseline_sharpe'] for r in results])
        
        total_trades = sum([r['num_trades'] for r in results])
        
        print(f"Average Portfolio Return: {total_portfolio_return*100:.2f}%")
        print(f"Average Baseline Return:  {total_baseline_return*100:.2f}%")
        print(f"Average Alpha:            {alpha*100:.2f}%")
        print(f"Average Portfolio Sharpe: {avg_portfolio_sharpe:.3f}")
        print(f"Average Baseline Sharpe:  {avg_baseline_sharpe:.3f}")
        print(f"Total Number of Trades:   {total_trades}")
        print(f"Number of Periods:        {len(results)}")
        
        # Success rate
        winning_periods = sum(1 for r in results if r['portfolio_return'] > r['baseline_return'])
        win_rate = winning_periods / len(results) * 100
        print(f"Win Rate:                 {win_rate:.1f}% ({winning_periods}/{len(results)} periods)")
        
        portfolio_returns = [r['portfolio_return'] for r in results]
        baseline_returns = [r['baseline_return'] for r in results]
        print(f"Portfolio Return Std:     {np.std(portfolio_returns)*100:.2f}%")
        print(f"Baseline Return Std:      {np.std(baseline_returns)*100:.2f}%")
        print(f"Best Period Return:       {max(portfolio_returns)*100:.2f}%")
        print(f"Worst Period Return:      {min(portfolio_returns)*100:.2f}%\n\n")
        
def main():
    """Main function to run the backtester."""
    parser = argparse.ArgumentParser(description='StockFormer Backtester')
    parser.add_argument('--data', type=str, default='period_splits.pkl', help='Path to period splits data')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--cost', type=float, default=0.001, help='Transaction cost (as decimal)')
    parser.add_argument('--output', type=str, default='backtest_results.png', help='Output plot filename')
    
    args = parser.parse_args()
    
    # Initialize backtester
    backtester = StockFormerBacktester(
        initial_capital=args.capital,
        cost=args.cost
    )
    
    # Run backtest
    results = backtester.run_backtest(
        data_file=args.data,
    )
    
    if results:
        # Print summary
        backtester.print_summary(results)
        
        # Plot results
        backtester.plot_results(results, args.output)
        
        # Save detailed results
        results_df = pd.DataFrame([{
            'period': r['period'],
            'portfolio_return': r['portfolio_return'],
            'baseline_return': r['baseline_return'],
            'alpha': r['portfolio_return'] - r['baseline_return'],
            'portfolio_sharpe': r['portfolio_sharpe'],
            'baseline_sharpe': r['baseline_sharpe'],
            'num_trades': r['num_trades']
        } for r in results])
        
        results_df.to_csv('backtest_summary.csv', index=False)

        print("\nBacktest completed.")
        
    else:
        print("Insufficient data for backtesting.")

if __name__ == "__main__":
    main()
