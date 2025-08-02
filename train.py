
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from StockFormer import create_compiled_stockformer
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os


def masked_mae(preds, labels, null_val=np.nan):
    """
    Compute mean absolute error with masking for missing values.
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

class MultiSupervisionLoss(nn.Module):
    """
    Custom loss combining regression (MAE) and classification (cross-entropy) for multi-task supervision.
    """
    def __init__(self, lambda_weight=1.0):
        super().__init__()
        self.weight = lambda_weight
        self.MAE_loss = nn.L1Loss()
        self.CE_loss = nn.CrossEntropyLoss()

    def forward(self, out, y_cla_batch, y_reg_batch):
        # Flatten outputs for loss calculation
        cla_pred = out["cla"].reshape(-1, 2)
        lcla_pred = out["lcla"].reshape(-1, 2)
        lreg_pred = out["lreg"].view(-1)
        reg_pred = out["reg"].view(-1)
        y_true_cla = (y_cla_batch.view(-1) == 1).long()
        y_true_reg = y_reg_batch.view(-1)

        # MAE loss for regression outputs
        reg_loss = masked_mae(lreg_pred, y_true_reg) + \
                   masked_mae(reg_pred, y_true_reg)

        # Cross-entropy loss for classification outputs
        cla_loss = self.CE_loss(lcla_pred, y_true_cla) + \
                   self.CE_loss(cla_pred, y_true_cla)

        # Combine losses
        total_loss = reg_loss + self.weight * cla_loss
        print(f"Loss - Regression: {reg_loss.item():.4f}, Classification: {cla_loss.item():.4f}")

        return total_loss


def train_period(period_data, model, optimizer, criterion, device, num_epochs, batch_size, 
                patience, model_path='stockformer_model.pth', scheduler=None, show_plot=False):
    """
    Train model on a single period with early stopping and live loss plotting.
    """
    # Convert to numpy arrays first
    X_train = np.array(period_data['training']['X'])
    Y_train = np.array(period_data['training']['Y'])
    Ts_train = np.array(period_data['training']['Ts'])
    X_val = np.array(period_data['validation']['X'])
    Y_val = np.array(period_data['validation']['Y'])
    Ts_val = np.array(period_data['validation']['Ts'])

    train_size = X_train.shape[0]
    val_size = X_val.shape[0]
    
    print(f"Period dataset - Train intervals: {train_size}, Val intervals: {val_size}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Initialize live plotting
    if show_plot:
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
            y_reg_batch = torch.tensor(Y_train_shuffled[i:end_idx][..., 0], dtype=torch.float32).to(device)
            y_cla_batch = torch.tensor(Y_train_shuffled[i:end_idx][..., 1], dtype=torch.long).to(device)
            ts_batch = torch.tensor(Ts_train_shuffled[i:end_idx], dtype=torch.float32).to(device)

            optimizer.zero_grad()
            
            # Forward pass
            out = model(x_batch, ts_batch)

            # Compute loss
            loss = criterion(out, y_cla_batch, y_reg_batch)

            # Backward pass
            loss.backward()

            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
                y_reg_batch = torch.tensor(Y_val[i:end_idx][..., 0], dtype=torch.float32).to(device)
                y_cla_batch = torch.tensor(Y_val[i:end_idx][..., 1], dtype=torch.long).to(device)
                ts_batch = torch.tensor(Ts_val[i:end_idx], dtype=torch.float32).to(device)
                
                out = model(x_batch, ts_batch)
                
                # Compute loss
                loss = criterion(out, y_cla_batch, y_reg_batch)
                    
                val_loss += loss.item()
                val_batches += 1
            
            if val_batches > 0:
                val_loss /= val_batches
                print(f"Validation Loss: {val_loss:.4f}")
                
                # Step scheduler with validation loss if provided
                if scheduler is not None:
                    pass
                    #scheduler.step()
                
                # Early stopping logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    torch.save(best_model_state, model_path)
                    patience_counter = 0
                    print(f"New best validation loss: {best_val_loss:.4f}, Model saved! Patience reset.")
                else:
                    patience_counter += 1
                    print(f"No improvement. Patience: {patience_counter}/{patience}")
                
                if show_plot:
                    # Update live plot
                    epochs_list.append(epoch + 1)
                    train_losses.append(avg_loss)
                    val_losses.append(val_loss)
                    update_plot()
                
                # Check for early stopping
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs. Best validation loss: {best_val_loss:.4f}")
                    break
            else:
                print("No validation batches available - skipping validation")

                if show_plot:
                    # Still update plot with training loss only
                    epochs_list.append(epoch + 1)
                    train_losses.append(avg_loss)
                    val_losses.append(float('nan'))  # Use NaN for missing validation loss
                    update_plot()

    # Load the best model state before finishing
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.4f}")

    if show_plot:
        plt.ioff()
        plt.close(fig)

    return best_val_loss

def train_on_periods(periods_dataset, model, optimizer, criterion, device, num_epochs, batch_size, model_path='stockformer_model.pth', reset_optimizer=True, weight_decay=0.1):
    """
    Train the model sequentially on all periods, resetting optimizer and adjusting learning rate/weight decay as needed.
    Loads the best model from the previous period (if available) and applies learning rate decay between periods.
    """
    # Store initial learning rate and weight decay for reset
    initial_lr = optimizer.param_groups[0]['lr']
    initial_wd = weight_decay
    base_patience = 15  # Define a base patience value
    
    period_files = sorted([
        os.path.join('training_periods', fname)
        for fname in os.listdir('training_periods')
        if fname.endswith('.pkl')
    ])
    for period_idx, period_file in enumerate(period_files):


        with open(period_file, 'rb') as f:
            period_data = pickle.load(f)['data']

        print(f"\nTraining on period {period_idx} with {len(period_data['training']['X'])} training samples and {len(period_data['validation']['X'])} validation samples")
        
        # Load best model from previous period (except for first period)
        if period_idx > 0:
            try:
                print(f"Loading best model from previous period...")
                best_state = torch.load(model_path, map_location=device)
                model.load_state_dict(best_state)
                print("Successfully loaded previous period's best model")
            except Exception as e:
                print(f"Warning: Could not load previous model: {e}")
                print("Continuing with current model state...")
        
        # Reset optimizer for each period with appropriate learning rate and weight decay
        if reset_optimizer:
            print("Resetting optimizer for new period...")
            
            # Learning rate and weight decay decay between periods
            if period_idx > 0:
                # Linear decay for learning rate and weight decay
                period_lr = initial_lr * (1.0 - 0.15 * period_idx)  # 15% reduction per period
                period_wd = initial_wd * (1.0 - 0.1 * period_idx)   # 10% reduction per period
                # Ensure minimum values
                period_lr = max(period_lr, 1e-4)
                period_wd = max(period_wd, 0.01)
                print(f"Adjusted learning rate: {initial_lr:.6f} → {period_lr:.6f}")
                print(f"Adjusted weight decay: {initial_wd:.6f} → {period_wd:.6f}")
            else:
                period_lr = initial_lr
                period_wd = initial_wd
            # Create new optimizer for this period
            optimizer = optim.Adam(model.parameters(), lr=period_lr, weight_decay=period_wd)

            # Optionally, create a learning rate scheduler for this period (currently disabled)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            #     optimizer,
            #     T_0=10,          # Number of epochs for the first restart
            #     T_mult=2,        # Multiplicative factor for T_0 after each restart
            #     eta_min=1e-6,    # Minimum learning rate
            #     verbose=True     # Print learning rate changes
            # )
            scheduler = None
        
        # Print current learning rate and weight decay at start of period
        current_lr = optimizer.param_groups[0]['lr']
        current_wd = optimizer.param_groups[0]['weight_decay']
        print(f"Starting period {period_idx} with learning rate: {current_lr:.6f}, weight decay: {current_wd:.6f}")
        
        # Pass scheduler to train_period (currently None)
        val_loss = train_period(period_data, model, optimizer, criterion, device, num_epochs, 
                               batch_size, base_patience, model_path, scheduler)
        
        del period_data  # Free memory after each period


def main():

    # Load config from YAML file
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Data and model path
    data_path = config.get('data', 'period_splits.pkl')
    model_path = config.get('model_path', 'stockformer_model.pth')

    # Training params
    train_params = config.get('train_params')
    epochs = train_params.get('epochs')
    batch_size = train_params.get('batch_size')
    device = train_params.get('device')
    reset_optimizer = train_params.get('reset_optimizer')
    learning_rate = train_params.get('initial_learning_rate')
    weight_decay = train_params.get('initial_weight_decay')
    cla_loss_weight = train_params.get('cla_loss_weight')

    # Model params
    model_params = config.get('model_params')
    seq_len = model_params.get('seq_len')
    pred_len = model_params.get('pred_len')
    num_features = model_params.get('num_features')
    d_model = model_params.get('d_model')
    num_heads = model_params.get('num_heads')
    dropout = model_params.get('dropout')
    pred_features = model_params.get('pred_features')

    # Load data
    with open('training_periods/period_split_0.pkl', 'rb') as f:
        data = pickle.load(f)

    lag = data['seq_len']
    lead = data['pred_len']
    # Sanity checks
    if lag != seq_len:
        raise ValueError("Sequence length mismatch between config and data.")
    if lead != pred_len:
        raise ValueError("Prediction length mismatch between config and data.")
    #period_splits = data['period_splits']
    tickers = data['tickers']
    num_stocks = len(tickers)

    print(f"Number of stocks: {num_stocks}, Number of features: {num_features}")

    model = create_compiled_stockformer(
        num_stocks=num_stocks,
        num_features=num_features,
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        pred_features=pred_features,
        device=device
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = MultiSupervisionLoss(cla_loss_weight)

    #print(f"Starting training on {len(period_splits)} periods...")
    print(f"Using device: {device}")
    print(f"Reset optimizer between periods: {reset_optimizer}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")

    # Train sequentially on periods dataset
    train_on_periods(None, model, optimizer, criterion, device,
                    epochs, batch_size, model_path, reset_optimizer, weight_decay)

    # Save final model along with config
    torch.save({'model_state_dict': model.state_dict(), 'config': config}, model_path)
    print(f"\nTraining completed! Final model saved to '{model_path}'")

if __name__ == "__main__":
    main()