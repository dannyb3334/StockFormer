import torch
import pywt
import torch.nn as nn

class DecouplingFlowLayer(nn.Module):
    """
    Wavelet-based decoupling layer for time series data.
    """
    
    def __init__(self, wavelet='haar', d_model=64, num_features=6):
        super(DecouplingFlowLayer, self).__init__()
        self.wavelet = wavelet
        self.num_features = num_features
        self.d_model = d_model
        # Linear layers to project low and high frequency components to d_model
        self.Wg = nn.Linear(num_features, d_model)
        self.Wh = nn.Linear(num_features, d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, num_stocks, num_features)
        batch_size, seq_len, num_stocks, num_features = x.shape

        # Rearrange x to (batch_size, num_stocks, num_features, seq_len)
        x_perm = x.permute(0, 2, 3, 1)
        # Flatten batch, stock, and feature dims for wavelet processing
        x_flat = x_perm.reshape(-1, seq_len)  # (batch_size*num_stocks*num_features, seq_len)

        # Lists to store low (cA) and high (cD) frequency components
        X_l, X_h = [], []
        for i in range(x_flat.shape[0]):
            # Perform 1D Discrete Wavelet Transform on each time series
            cA, cD = pywt.dwt(x_flat[i].cpu().numpy(), self.wavelet)
            X_l.append(torch.tensor(cA, dtype=x.dtype, device=x.device))
            X_h.append(torch.tensor(cD, dtype=x.dtype, device=x.device))
        X_l = torch.stack(X_l, dim=0)
        X_h = torch.stack(X_h, dim=0)

        # Inverse DWT to upsample back to original sequence length
        X_l_up, X_h_up = [], []
        for i in range(X_l.shape[0]):
            rec_l = pywt.idwt(X_l[i].cpu().numpy(), None, self.wavelet)
            rec_h = pywt.idwt(None, X_h[i].cpu().numpy(), self.wavelet)
            rec_l = torch.tensor(rec_l, dtype=x.dtype, device=x.device)
            rec_h = torch.tensor(rec_h, dtype=x.dtype, device=x.device)
            if rec_l.shape[0] < seq_len:
                rec_l = torch.cat([rec_l, rec_l.new_zeros(seq_len - rec_l.shape[0])])
            if rec_h.shape[0] < seq_len:
                rec_h = torch.cat([rec_h, rec_h.new_zeros(seq_len - rec_h.shape[0])])
            X_l_up.append(rec_l[:seq_len])
            X_h_up.append(rec_h[:seq_len])
        X_l_up = torch.stack(X_l_up, dim=0)
        X_h_up = torch.stack(X_h_up, dim=0)
        # Reshape back to (batch_size, seq_len, num_stocks, num_features)
        X_l = X_l_up.view(batch_size, num_stocks, num_features, seq_len).permute(0, 3, 1, 2)
        X_h = X_h_up.view(batch_size, num_stocks, num_features, seq_len).permute(0, 3, 1, 2)

        return X_l, X_h

class DualFrequencyEncoder(nn.Module):
    """
    Dual-frequency spatiotemporal encoder for time series data.
    """

    def __init__(self, d_model, num_features):
        super(DualFrequencyEncoder, self).__init__()
        self.temporal_attention = nn.MultiheadAttention(embed_dim=num_features*2, num_heads=4, batch_first=True)
        self.dilated_conv = nn.Conv1d(in_channels=num_features*2, out_channels=d_model, kernel_size=3, dilation=2)
        self.relu = nn.ReLU()

    def forward(self, X_l_cat, X_h_cat, ts):
        # Temporal Attention on low-frequency components
        batch_size, seq_len, num_stocks, num_features = X_l_cat.shape
        X_l_flat = X_l_cat.view(batch_size * num_stocks, seq_len, num_features)
        X_tatt, _ = self.temporal_attention(X_l_flat, X_l_flat, X_l_flat)
        X_tatt = X_tatt.view(batch_size, seq_len, num_stocks, num_features)

        # Dilated Causal Convolution on high-frequency components
        X_h_flat = X_h_cat.permute(0, 2, 3, 1).reshape(batch_size * num_stocks, num_features, seq_len)
        X_conv = self.dilated_conv(X_h_flat)
        X_conv = self.relu(X_conv)
        X_conv = X_conv.reshape(batch_size, num_stocks, -1, seq_len).permute(0, 3, 1, 2)
        # TODO: Time Slot and Struc2Vec Graph Attention Layer
        # ts: (batch_size, seq_len) unix timestamps for sequence X
        
        return X_l_gat, X_h_gat
        
class StockFormer(nn.Module):
    """
    StockFormer model for time series forecasting
    """

    def __init__(self, num_stocks, num_features, d_model=64, nhead=4, num_layers=2, pred_features=[0, 1]):
        super(StockFormer, self).__init__()
        self.num_stocks = num_stocks
        self.num_features = num_features
        self.pred_features = pred_features

        # Wavelet-based decoupling layer
        self.decouple = DecouplingFlowLayer()

        # Dual-frequency spatiotemporal encoder
        self.dual_freq_encoder = DualFrequencyEncoder(d_model, num_features)

        # Linear projection from input features to model dimension
        self.input_proj = nn.Linear(num_features * 2, d_model)

        # Transformer encoder for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection: predicts 2 timesteps ahead for selected features
        self.output_proj = nn.Linear(d_model, len(pred_features) * 2)

    def forward(self, x, ts):
        # x: (batch_size, seq_len, num_stocks, num_features)
        # Decompose input into low and high frequency components, then concat
        X_l, X_h = self.decouple(x)  # Both: (batch_size, seq_len, num_stocks, d_model)

        #X_l : long-term trends
        #X_h : short-term fluctuations
        
        # Concatenate along feature dimension to combine both components
        X_l_cat = torch.cat([x, X_l], dim=-1)  # (batch_size, seq_len, num_stocks, num_features*2)
        X_h_cat = torch.cat([x, X_h], dim=-1)  # (batch_size, seq_len, num_stocks, num_features*2)
    
        X_l_gat, X_h_gat= self.dual_freq_encoder(X_l_cat, X_h_cat, ts)
        # TODO: Implement Dual-Frequency Fusion Decoder

        return out