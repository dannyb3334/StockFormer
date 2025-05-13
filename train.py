import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from StockFormer import StockFormer
from eval import metrics
import numpy as np
import argparse

def inverse_transform(y, mean, std):
    # y: (batch, lead, num_stocks, num_targets)
    # mean, std: (num_stocks, num_targets)
    return y * std + mean

def train_one_period(period_data, model, optimizer, criterion, device, num_epochs=10, batch_size=16):
    X_train, Y_train, Ts_train = period_data['training']['X'], period_data['training']['Y'], period_data['training']['Ts']
    X_val, Y_val, Ts_val = period_data['validation']['X'], period_data['validation']['Y'], period_data['validation']['Ts']
    mean, std = np.array(period_data['mean']), np.array(period_data['std'])

    # Convert to torch tensors
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32).to(device)
    Y_train = torch.tensor(np.array(Y_train), dtype=torch.float32).to(device)
    Ts_train = torch.tensor(np.array(Ts_train), dtype=torch.float32).to(device)

    X_val = torch.tensor(np.array(X_val), dtype=torch.float32).to(device)
    Y_val = torch.tensor(np.array(Y_val), dtype=torch.float32).to(device)
    Ts_val = torch.tensor(np.array(Ts_val), dtype=torch.float32).to(device)

    train_size = X_train.shape[0]
    val_size = X_val.shape[0]

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(train_size)
        epoch_loss = 0
        for i in range(0, train_size, batch_size):
            idx = perm[i:i+batch_size]
            x_batch = X_train[idx]  # (batch, lag, num_stocks, num_features)
            y_batch = Y_train[idx]  # (batch, lead, num_stocks, num_targets)
            ts_batch = Ts_train[idx]  # (batch,)

            optimizer.zero_grad()
            # Only use the last lag for prediction, adjust as needed
            out = model(x_batch, ts_batch)
            # out: (batch, num_stocks, num_targets * lead)
            # Reshape y_batch to match out
            y_true = y_batch[:, 0, :, :].reshape(out.shape)
            loss = criterion(out, y_true)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/(train_size//batch_size):.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i in range(0, val_size, batch_size):
                x_batch = X_val[i:i+batch_size]
                y_batch = Y_val[i:i+batch_size]
                ts_batch = Ts_val[i:i+batch_size]
                out = model(x_batch, ts_batch)
                y_true = y_batch[:, 0, :, :].reshape(out.shape)
                val_loss += criterion(out, y_true).item()
            val_loss /= (val_size//batch_size)
            print(f"Validation Loss: {val_loss:.4f}")

            # Save checkpoint if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'checkpoint_period_{period_data["period"]}.pth')
                print(f"Saved checkpoint for period {period_data['period']} with validation loss: {best_val_loss:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='period_splits.pkl', help='Pickle file with period splits')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load data
    with open(args.data, 'rb') as f:
        period_splits = pickle.load(f)

    # Get model dimensions from data
    first_period = next(iter(period_splits.values()))
    X_sample = np.array(first_period['training']['X'])[0]  # (lag, num_stocks, num_features)
    num_stocks = X_sample.shape[1]
    num_features = X_sample.shape[2]

    model = StockFormer(num_stocks=num_stocks, num_features=num_features).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for period_idx, period_data in period_splits.items():
        print(f"\nTraining on period {period_idx}")
        train_one_period(period_data, model, optimizer, criterion, args.device, num_epochs=args.epochs, batch_size=args.batch_size)
        # TODO: Test set
if __name__ == "__main__":
    main()