import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from StockFormer import StockFormer
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time
import os

def inverse_transform(y, mean, std):
    # y: (batch, lead, num_stocks, num_targets)
    # mean, std: (num_stocks, num_targets)
    return y * std + mean

def combine_all_periods(period_splits):
    """Combine all periods into single training and validation sets"""
    all_X_train, all_Y_train, all_Ts_train = [], [], []
    all_X_val, all_Y_val, all_Ts_val = [], [], []
    
    for period_idx, period_data in period_splits.items():
        # Training data
        all_X_train.extend(period_data['training']['X'])
        all_Y_train.extend(period_data['training']['Y'])
        all_Ts_train.extend(period_data['training']['Ts'])
        
        # Validation data
        all_X_val.extend(period_data['validation']['X'])
        all_Y_val.extend(period_data['validation']['Y'])
        all_Ts_val.extend(period_data['validation']['Ts'])
    
    return {
        'X_train': all_X_train, 'Y_train': all_Y_train, 'Ts_train': all_Ts_train,
        'X_val': all_X_val, 'Y_val': all_Y_val, 'Ts_val': all_Ts_val
    }

def train_model(combined_data, model, optimizer, criterion, device, model_path='stockformer_model.pth', num_epochs=200, batch_size=2048):
    """Train model on combined dataset"""
    # Convert to numpy arrays first
    X_train = np.array(combined_data['X_train'])
    Y_train = np.array(combined_data['Y_train'])
    Ts_train = np.array(combined_data['Ts_train'])
    X_val = np.array(combined_data['X_val'])
    Y_val = np.array(combined_data['Y_val'])
    Ts_val = np.array(combined_data['Ts_val'])

    train_size = X_train.shape[0]
    val_size = X_val.shape[0]
    
    print(f"Combined dataset - Train intervals: {train_size}, Val intervals: {val_size}")
    
    best_val_loss = float('inf')
    
    # Initialize live plotting
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    train_losses = []
    val_losses = []
    epochs_list = []
    
    # Set up the plot
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    train_line, = ax.plot([], [], 'b-', label='Training Loss', linewidth=2)
    val_line, = ax.plot([], [], 'r-', label='Validation Loss', linewidth=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    def update_plot():
        if epochs_list:
            train_line.set_data(epochs_list, train_losses)
            val_line.set_data(epochs_list, val_losses)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)
    
    for epoch in range(num_epochs):
        model.train()
        
        # Shuffle training data
        perm = np.random.permutation(train_size)
        X_train_shuffled = X_train[perm]
        Y_train_shuffled = Y_train[perm]
        Ts_train_shuffled = Ts_train[perm]
        
        epoch_loss = 0
        num_batches = 0
        
        # Training loop
        for i in range(0, train_size, batch_size):
            end_idx = min(train_size, i + batch_size)
            
            # Convert batch to tensors
            x_batch = torch.tensor(X_train_shuffled[i:end_idx], dtype=torch.float32).to(device)
            y_batch = torch.tensor(Y_train_shuffled[i:end_idx], dtype=torch.float32).to(device)
            ts_batch = torch.tensor(Ts_train_shuffled[i:end_idx], dtype=torch.long).to(device)

            optimizer.zero_grad()
            
            # Forward pass
            out = model.predict(x_batch, ts_batch)  # (batch, num_stocks, 2)
            y_true = y_batch[:, 0, :, :].view(out.shape)  # (batch, num_stocks, 2)
            
            loss = criterion(out, y_true)
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_batches = 0
            
            for i in range(0, val_size, batch_size):
                end_idx = min(val_size, i + batch_size)
                
                x_batch = torch.tensor(X_val[i:end_idx], dtype=torch.float32).to(device)
                y_batch = torch.tensor(Y_val[i:end_idx], dtype=torch.float32).to(device)
                ts_batch = torch.tensor(Ts_val[i:end_idx], dtype=torch.long).to(device)
                
                out = model.predict(x_batch, ts_batch)
                y_true = y_batch[:, 0, :, :].view(out.shape)
                val_loss += criterion(out, y_true).item()
                val_batches += 1
            
            if val_batches > 0:
                val_loss /= val_batches
                print(f"Validation Loss: {val_loss:.4f}")

                # Save model only when validation improves
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_path)
                    print(f"New best validation loss: {best_val_loss:.4f}, Model saved!")
                
                # Update live plot
                epochs_list.append(epoch + 1)
                train_losses.append(avg_loss)
                val_losses.append(val_loss)
                update_plot()
            else:
                print("No validation batches available - skipping validation")
                # Still update plot with training loss only
                epochs_list.append(epoch + 1)
                train_losses.append(avg_loss)
                val_losses.append(float('nan'))  # Use NaN for missing validation loss
                update_plot()
    
    # Keep the plot open at the end
    plt.ioff()
    plt.show()
    print("Training completed! Close the plot window to continue.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='period_splits.pkl', help='Pickle file with period splits')
    parser.add_argument('--epochs', type=int, default=200) # StockFormer default is 200 epochs
    parser.add_argument('--batch_size', type=int, default=2048) # StockFormer default is 2048
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_path', type=str, default='stockformer_model.pth', help='Path to save/load model')
    args = parser.parse_args()

    # Load data
    with open(args.data, 'rb') as f:
        period_splits = pickle.load(f)

    # Get model dimensions from data
    first_period = next(iter(period_splits.values()))
    X_sample = np.array(first_period['training']['X'])[0]  # (lag, num_stocks, num_features)
    num_stocks = X_sample.shape[1]
    num_features = X_sample.shape[2]
    print(f"Number of stocks: {num_stocks}, Number of features: {num_features}")

    model = StockFormer(num_stocks=num_stocks, num_features=num_features).to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print(f"Starting training on {len(period_splits)} periods...")
    print(f"Using device: {args.device}")

    # Combine all periods into single dataset
    combined_data = combine_all_periods(period_splits)
    print(f"Combined all periods into single dataset")
    
    # Train on combined dataset
    train_model(combined_data, model, optimizer, criterion, args.device, args.model_path, num_epochs=args.epochs, batch_size=args.batch_size)
    
    # Save final model
    torch.save(model.state_dict(), args.model_path)
    print(f"\nTraining completed! Final model saved to '{args.model_path}'")
    print(f"Ready for backtesting with: python backtest_engine.py")
if __name__ == "__main__":
    main()