import torch
import pywt
import torch.nn as nn
import torch.nn.functional as F
import math

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
        x_perm = x.permute(0, 2, 3, 1)  # (batch_size, num_stocks, num_features, seq_len)
        # Flatten batch, stock, and feature dims for wavelet processing
        x_flat = x_perm.reshape(-1, seq_len)  # (batch_size*num_stocks*num_features, seq_len)

        # Lists to store low (cA) and high (cD) frequency components
        X_l, X_h = [], []
        for i in range(x_flat.shape[0]):
            # Perform 1D Discrete Wavelet Transform on each time series
            cA, cD = pywt.dwt(x_flat[i].cpu().numpy(), self.wavelet)
            X_l.append(torch.tensor(cA, dtype=x.dtype, device=x.device))
            X_h.append(torch.tensor(cD, dtype=x.dtype, device=x.device))
        X_l = torch.stack(X_l, dim=0)  # (batch_size*num_stocks*num_features, seq_len//2)
        X_h = torch.stack(X_h, dim=0)  # (batch_size*num_stocks*num_features, seq_len//2)

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
        X_l_up = torch.stack(X_l_up, dim=0)  # (batch_size*num_stocks*num_features, seq_len)
        X_h_up = torch.stack(X_h_up, dim=0)  # (batch_size*num_stocks*num_features, seq_len)
        # Reshape back to (batch_size, seq_len, num_stocks, num_features)
        X_l = X_l_up.view(batch_size, num_stocks, num_features, seq_len).permute(0, 3, 1, 2)  # (batch_size, seq_len, num_stocks, num_features)
        X_h = X_h_up.view(batch_size, num_stocks, num_features, seq_len).permute(0, 3, 1, 2)  # (batch_size, seq_len, num_stocks, num_features)

        return X_l, X_h  # both: (batch_size, seq_len, num_stocks, num_features)

class DualFrequencyEncoder(nn.Module):
    def __init__(self, seq_len, d_model, num_features):
        super(DualFrequencyEncoder, self).__init__()
        # Temporal attention for low-frequency components
        self.temporal_attention = nn.MultiheadAttention(embed_dim=num_features * 2, num_heads=4, batch_first=True)
        self.dilated_conv = nn.Conv1d(in_channels=num_features * 2, out_channels=d_model, kernel_size=3, dilation=2)
        self.relu = nn.ReLU()

        # Positional encoding (fixed)
        # For decoupled low frequency, along sequence length
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', pe)

        # Time embedding of time slots
        self.num_time_slots = 252
        self.time_embedding = nn.Linear(self.num_time_slots, d_model)

        # Spatial embedding
        self.spatial_embedding = nn.Parameter(torch.randn(1, self.num_stocks, d_model))  # learnable N × d_model

    def embed_time_slots(self, ts_index):
        # ts_index: (batch_size, seq_len), values of [0, 251]
        one_hot = F.one_hot(ts_index, num_classes=self.num_time_slots).float()  # (batch, seq, 252)
        time_embed = self.time_embedding(one_hot)  # (batch, seq, d_model)
        return time_embed  # (batch, seq, d_model)

    def forward(self, X_l_cat, X_h_cat, ts_index):
        # X_l_cat: (batch_size, seq_len, num_stocks, num_features*2)
        # X_h_cat: (batch_size, seq_len, num_stocks, num_features*2)
        # ts_index: (batch_size, seq_len)
        batch_size, seq_len, num_stocks, num_features = X_l_cat.shape

        # Temporal Attention on low-frequency
        # Flatten for multi-head attention
        X_l_flat = X_l_cat.view(batch_size * num_stocks, seq_len, num_features)  # (batch_size*num_stocks, seq_len, num_features*2)
        X_l_flat = X_l_flat + self.positional_encoding[:seq_len, :]  # (batch_size*num_stocks, seq_len, num_features*2)
        X_tatt, _ = self.temporal_attention(X_l_flat, X_l_flat, X_l_flat)  # (batch_size*num_stocks, seq_len, num_features*2)
        X_tatt = X_tatt.view(batch_size, seq_len, num_stocks, num_features)  # (batch_size, seq_len, num_stocks, num_features*2)

        # Dilated Conv on high-frequency
        X_h_flat = X_h_cat.permute(0, 2, 3, 1).reshape(batch_size * num_stocks, num_features, seq_len)  # (batch_size*num_stocks, num_features*2, seq_len)
        X_conv = self.dilated_conv(X_h_flat)  # (batch_size*num_stocks, d_model, L_out)
        X_conv = self.relu(X_conv)  # (batch_size*num_stocks, d_model, L_out)
        X_conv = X_conv.reshape(batch_size, num_stocks, -1, seq_len).permute(0, 3, 1, 2)  # (batch_size, seq_len, num_stocks, d_model)

        # Temporal Graph Embedding
        time_embed = self.embed_time_slots(ts_index)  # (batch, seq, d_model)
        time_embed = time_embed.unsqueeze(2).repeat(1, 1, num_stocks, 1)  # (batch, seq, stocks, d_model)

        # Struc2Vec Graph Embedding
        # TODO: Implement Struc2Vec Graph Embedding

        # Placeholder for outputs
        X_l_gat = X_tatt  # (batch_size, seq_len, num_stocks, num_features*2)
        X_h_gat = X_conv  # (batch_size, seq_len, num_stocks, d_model)

        return X_l_gat, X_h_gat

class DualFrequencyFusionDecoder(nn.Module):
    def __init__(self, seq_len, d_model, num_features):
        super(DualFrequencyFusionDecoder, self).__init__()

        # Predictors (i.e fully connected layers)
        self.predictor_l = nn.Linear(d_model, d_model)
        self.predictor_h = nn.Linear(d_model, d_model)

        # Positional encoding (fixed)
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', pe.unsqueeze(0))

        # Attention layers
        self.attn_self = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.attn_cross = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

        # Final outputs from predictors
        self.reg_low = nn.Linear(d_model, 1)
        self.cla_low = nn.Linear(d_model, 2)
        self.reg_fused = nn.Linear(d_model, 1)
        self.cla_fused = nn.Linear(d_model, 2)

    def forward(self, X_l_gat, X_h_gat):
        # X_l_gat: (batch_size, seq_len, num_stocks, d_model)
        # X_h_gat: (batch_size, seq_len, num_stocks, d_model)
        batch_size, seq_len, num_stocks, d_model = X_l_gat.shape

        # Flatten for predictor and attention
        Y_l = X_l_gat.view(batch_size * num_stocks, seq_len, d_model)  # (batch_size*num_stocks, seq_len, d_model)
        Y_h = X_h_gat.view(batch_size * num_stocks, seq_len, d_model)  # (batch_size*num_stocks, seq_len, d_model)

        # Predictors
        # Format (Q: Query, K: Key, V: Value), for low, high
        Q_l = self.predictor_l(Y_l)  # (batch_size*num_stocks, seq_len, d_model)
        K_h = self.predictor_h(Y_h)  # (batch_size*num_stocks, seq_len, d_model)

        K_l = Q_l
        V_l = Q_l

        V_h = K_h

        # Predictor for low-frequency path
        Y_l = self.predictor_l(Q_l) # (batch_size*num_stocks, seq_len, d_model)
        Y_l_reg = self.reg_low(Y_l).view(batch_size, seq_len, num_stocks) # (batch_size, seq_len, num_stocks)
        Y_l_cla = F.softmax(self.cla_low(Y_l), dim=-1).view(batch_size, seq_len, num_stocks, 2) # (batch_size, seq_len, num_stocks, 2)

        # Predictor for high-frequency path
        Y_h = self.predictor_h(K_h) # (batch_size*num_stocks, seq_len, d_model)

        # Fusion Attention
        attn_self, _ = self.attn_self(Q_l, K_l, V_l)  # self-attention on low
        attn_cross, _ = self.attn_cross(Q_l, K_h, V_h)  # cross attention

        fused = attn_self + attn_cross  # (B*N, T, D)
        fused = fused.view(batch_size, seq_len, num_stocks, d_model)

        # Final predictions on fused output
        Y_reg = self.reg_fused(fused).squeeze(-1) # (batch_size, seq_len, num_stocks)
        Y_cla = F.softmax(self.cla_fused(fused), dim=-1)  # (batch_size, seq_len, num_stocks, 2)

        return {
            "reg": Y_reg,
            "cla": Y_cla,
            "lreg": Y_l_reg,
            "lcla": Y_l_cla
        }
class StockFormer(nn.Module):
    """.positional_encoding[:seq_len, :]  # Example addition
    StockFormer model for time series forecasting
    """

    def __init__(self, num_stocks, seq_len, num_features, d_model=64, num_heads=4, num_layers=2, pred_features=[0, 1]):
        super(StockFormer, self).__init__()
        self.num_stocks = num_stocks
        self.num_features = num_features
        self.pred_features = pred_features
        self.seq_len = seq_len

        # Wavelet-based decoupling layer
        self.decouple = DecouplingFlowLayer()

        # Dual-frequency spatiotemporal encoder
        self.dual_freq_encoder = DualFrequencyEncoder(seq_len, d_model, num_features)

        # Dual-frequency fusion decoder
        self.dual_freq_fusion_decoder = DualFrequencyFusionDecoder(seq_len, d_model, num_features)

        # Linear projection from input features to model dimension
        self.input_proj = nn.Linear(num_features * 2, d_model)

        # Transformer encoder for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection: predicts 2 timesteps ahead for selected features
        self.output_proj = nn.Linear(d_model, len(pred_features) * 2)
    
    def compute_loss(preds, y_true_reg, y_true_cla, lambda_cla=1.0):
        # preds["reg"]: (batch_size, seq_len, num_stocks)
        # preds["lreg"]: (batch_size, seq_len, num_stocks)
        # preds["cla"]: (batch_size, seq_len, num_stocks, 2)
        # preds["lcla"]: (batch_size, seq_len, num_stocks, 2)
        # y_true_reg: (batch_size, seq_len, num_stocks)
        # y_true_cla: (batch_size, seq_len, num_stocks)
        l1 = F.l1_loss(preds["reg"], y_true_reg)
        l2 = F.l1_loss(preds["lreg"], y_true_reg)
        c1 = F.cross_entropy(preds["cla"].permute(0, 2, 3, 1).reshape(-1, 2), y_true_cla.view(-1))
        c2 = F.cross_entropy(preds["lcla"].permute(0, 2, 3, 1).reshape(-1, 2), y_true_cla.view(-1))
        return l1 + l2 + lambda_cla * (c1 + c2)

    def forward(self, x, ts):
        # x: (batch_size, seq_len, num_stocks, num_features)
        # ts: (batch_size, seq_len) unix timestamps for sequence x
        # Decompose input into low and high frequency components, then concat
        X_l, X_h = self.decouple(x)  # Both: (batch_size, seq_len, num_stocks, d_model)

        # X_l : long-term trends (batch_size, seq_len, num_stocks, num_features)
        # X_h : short-term fluctuations (batch_size, seq_len, num_stocks, num_features)
        
        # Concatenate along feature dimension to combine both components
        X_l_cat = torch.cat([x, X_l], dim=-1)  # (batch_size, seq_len, num_stocks, num_features*2)
        X_h_cat = torch.cat([x, X_h], dim=-1)  # (batch_size, seq_len, num_stocks, num_features*2)
    
        X_l_gat, X_h_gat = self.dual_freq_encoder(X_l_cat, X_h_cat, ts)  # X_l_gat: (batch_size, seq_len, num_stocks, d_model), X_h_gat: (batch_size, seq_len, num_stocks, d_model)
        predictions = self.dual_freq_fusion_decoder(X_l_gat, X_h_gat)    # dict of shapes as above

        return predictions