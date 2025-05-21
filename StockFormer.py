import torch
import pywt
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
import math

class DecouplingFlowLayer(nn.Module):
    """
    Wavelet-based decoupling layer that decomposes only the stock return series into low and high frequency components.
    """
    def __init__(self, wavelet='haar', d_model=64, num_features=362, return_index=0):
        super(DecouplingFlowLayer, self).__init__()
        self.wavelet = wavelet
        self.d_model = d_model
        self.num_features = num_features
        self.return_index = return_index  # Index of the return feature (default 0)
        # Linear layers to project to d_model dimensions
        self.Wg = nn.Linear(num_features, d_model)
        self.Wh = nn.Linear(num_features, d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, num_stocks, num_features)
        batch_size, seq_len, num_stocks, num_features = x.shape

        # Extract return series (assumed at return_index)
        return_series = x[:, :, :, self.return_index]  # (batch_size, seq_len, num_stocks)
        return_series = return_series.permute(0, 2, 1)  # (batch_size, num_stocks, seq_len)
        return_series_flat = return_series.reshape(-1, seq_len)  # (batch_size*num_stocks, seq_len)

        # Apply DWT (pywt) to return series only
        X_l, X_h = [], []
        for i in range(return_series_flat.shape[0]):
            cA, cD = pywt.dwt(return_series_flat[i].cpu().numpy(), self.wavelet)
            X_l.append(torch.tensor(cA, dtype=x.dtype, device=x.device))
            X_h.append(torch.tensor(cD, dtype=x.dtype, device=x.device))
        X_l = torch.stack(X_l, dim=0)  # (batch_size*num_stocks, seq_len//2)
        X_h = torch.stack(X_h, dim=0)  # (batch_size*num_stocks, seq_len//2)

        # Upsample using IDWT (pywt)
        X_l_up, X_h_up = [], []
        for i in range(X_l.shape[0]):
            rec_l = pywt.idwt(X_l[i].cpu().numpy(), None, self.wavelet)
            rec_h = pywt.idwt(None, X_h[i].cpu().numpy(), self.wavelet)
            rec_l = torch.tensor(rec_l, dtype=x.dtype, device=x.device)
            rec_h = torch.tensor(rec_h, dtype=x.dtype, device=x.device)
            # Pad or truncate to match seq_len
            if rec_l.shape[0] < seq_len:
                rec_l = torch.cat([rec_l, rec_l.new_zeros(seq_len - rec_l.shape[0])])
            if rec_h.shape[0] < seq_len:
                rec_h = torch.cat([rec_h, rec_h.new_zeros(seq_len - rec_h.shape[0])])
            X_l_up.append(rec_l[:seq_len])
            X_h_up.append(rec_h[:seq_len])
        X_l_up = torch.stack(X_l_up, dim=0)  # (batch_size*num_stocks, seq_len)
        X_h_up = torch.stack(X_h_up, dim=0)  # (batch_size*num_stocks, seq_len)

        # Reshape back to (batch_size, seq_len, num_stocks)
        X_l = X_l_up.view(batch_size, num_stocks, seq_len).permute(0, 2, 1)  # (batch_size, seq_len, num_stocks)
        X_h = X_h_up.view(batch_size, num_stocks, seq_len).permute(0, 2, 1)  # (batch_size, seq_len, num_stocks)

        # Replace return feature in original input
        X_l_full = x.clone()
        X_l_full[:, :, :, self.return_index] = X_l  # (batch_size, seq_len, num_stocks, num_features)
        X_h_full = x.clone()
        X_h_full[:, :, :, self.return_index] = X_h  # (batch_size, seq_len, num_stocks, num_features)

        # Project to d_model dimensions
        X_l_proj = self.Wg(X_l_full.view(-1, num_features)).view(batch_size, seq_len, num_stocks, self.d_model)
        X_h_proj = self.Wh(X_h_full.view(-1, num_features)).view(batch_size, seq_len, num_stocks, self.d_model)

        return X_l_proj, X_h_proj  # Both: (batch_size, seq_len, num_stocks, d_model)

class GraphAttentionLayer(nn.Module): pass
class DualFrequencyEncoder(nn.Module):
    def __init__(self, seq_len, d_model, num_stocks, num_heads):
        super(DualFrequencyEncoder, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_stocks = num_stocks

        # Temporal attention for low-frequency components
        self.temporal_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        # Dilated convolution with padding to preserve sequence length
        self.dilated_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, dilation=2, padding=2)
        self.relu = nn.ReLU()

        # Positional encoding
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', pe)

        # Time embedding
        self.num_time_slots = 252
        self.time_embedding = nn.Linear(self.num_time_slots, d_model)

        # Spatial embedding
        self.spatial_embedding = nn.Parameter(torch.randn(num_stocks, d_model))

        # GraphAttentionLayer 
        self.gat_l = GraphAttentionLayer(d_model, num_heads=num_heads)
        self.gat_h = GraphAttentionLayer(d_model, num_heads=num_heads)

    def embed_time_slots(self, ts_index):
        # ts_index: (batch_size, seq_len), values in [0, 251]
        one_hot = F.one_hot(ts_index, num_classes=self.num_time_slots).float()  # (batch, seq, 252)
        time_embed = self.time_embedding(one_hot)  # (batch, seq, d_model)
        return time_embed  # (batch, seq, d_model)

    def embed_spatial(self, returns):
        # returns: (batch_size, seq_len, num_stocks)
        pass
        
    def compute_spearman_adj(self, returns):
        # returns: (batch_size, seq_len, num_stocks)
        pass
       

    def forward(self, X_l, X_h, ts_index, returns):
        # X_l, X_h: (batch_size, seq_len, num_stocks, d_model)
        # ts_index: (batch_size, seq_len)
        # returns: (batch_size, seq_len, num_stocks)
        batch_size, seq_len, num_stocks, d_model = X_l.shape

        # Temporal Attention on low-frequency
        X_l_flat = X_l.view(batch_size * num_stocks, seq_len, d_model)
        X_tatt, _ = self.temporal_attention(X_l_flat, X_l_flat, X_l_flat)  # (batch_size*num_stocks, seq_len, d_model)
        X_tatt = X_tatt.view(batch_size, seq_len, num_stocks, d_model)

        # Dilated Conv on high-frequency
        X_h_perm = X_h.permute(0, 2, 3, 1)  # (batch_size, num_stocks, d_model, seq_len)
        X_h_flat = X_h_perm.reshape(batch_size * num_stocks, d_model, seq_len)
        X_conv = self.dilated_conv(X_h_flat)  # (batch_size*num_stocks, d_model, seq_len)
        X_conv = self.relu(X_conv)
        X_conv = X_conv.view(batch_size, num_stocks, d_model, seq_len).permute(0, 3, 1, 2)  # (batch_size, seq_len, num_stocks, d_model)
        
        # Temporal Graph Embedding
        p_tem = self.embed_time_slots(ts_index)  # (batch, seq, d_model)
        p_tem = p_tem.unsqueeze(2).repeat(1, 1, num_stocks, 1)  # (batch, seq, num_stocks, d_model)

        # Spatial Embedding
        p_spa = self.embed_spatial(returns)  # (batch, seq, num_stocks, d_model)
        p_spa = p_spa.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # Combine embeddings
        X_l = X_tatt + p_tem + p_spa
        X_h = X_conv + p_tem + p_spa

        # Self-Attention (simplified, should be GAT)
        X_l_gat = self.gat_l(X_l)  # (batch, seq, num_stocks, d_model)
        X_h_gat = self.gat_h(X_h)  # (batch, seq, num_stocks, d_model)

        return X_l_gat, X_h_gat

class DualFrequencyFusionDecoder(nn.Module):
    def __init__(self, seq_len, d_model, num_heads):
        super(DualFrequencyFusionDecoder, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Predictors
        self.predictor_l = nn.Linear(d_model, d_model)
        self.predictor_h = nn.Linear(d_model, d_model)

        # Positional encoding
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', pe.unsqueeze(0))

        # Attention layers
        self.attn_self = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.attn_cross = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)

        # Output layers
        self.reg_low = nn.Linear(d_model, 1)
        self.cla_low = nn.Linear(d_model, 2)
        self.reg_fused = nn.Linear(d_model, 1)
        self.cla_fused = nn.Linear(d_model, 2)

    def forward(self, X_l_gat, X_h_gat) -> dict:
        # X_l_gat, X_h_gat: (batch_size, seq_len, num_stocks, d_model)
        batch_size, seq_len, num_stocks, d_model = X_l_gat.shape

        # Flatten for processing
        Y_l = X_l_gat.view(batch_size * num_stocks, seq_len, d_model)
        Y_h = X_h_gat.view(batch_size * num_stocks, seq_len, d_model)

        # Predictors
        pred_l = self.predictor_l(Y_l)  # (batch_size*num_stocks, seq_len, d_model)
        pred_h = self.predictor_h(Y_h)  # (batch_size*num_stocks, seq_len, d_model)

        # Add positional encoding
        input_l = pred_l + self.positional_encoding[:, :seq_len, :]
        input_h = pred_h + self.positional_encoding[:, :seq_len, :]

        # Fusion Attention
        attn_self_out, _ = self.attn_self(input_l, input_l, input_l)
        attn_cross_out, _ = self.attn_cross(input_l, input_h, input_h)
        output = attn_self_out + attn_cross_out  # (batch_size*num_stocks, seq_len, d_model)

        # Low-frequency outputs
        Y_l_reg = self.reg_low(pred_l).view(batch_size, seq_len, num_stocks)  # (batch_size, seq_len, num_stocks)
        Y_l_cla = F.softmax(self.cla_low(pred_l), dim=-1).view(batch_size, seq_len, num_stocks, 2)  # (batch_size, seq_len, num_stocks, 2)

        # Fused outputs
        fused = output.view(batch_size, seq_len, num_stocks, d_model)
        Y_reg = self.reg_fused(fused).squeeze(-1)  # (batch_size, seq_len, num_stocks)
        Y_cla = F.softmax(self.cla_fused(fused), dim=-1)  # (batch_size, seq_len, num_stocks, 2)

        return {
            "reg": Y_reg,
            "cla": Y_cla,
            "lreg": Y_l_reg,
            "lcla": Y_l_cla
        }

class StockFormer(nn.Module):
    """
    StockFormer model for time series forecasting.
    """
    def __init__(self, num_stocks, seq_len, num_features, d_model=64, num_heads=4, pred_features=[0, 1]):
        super(StockFormer, self).__init__()
        self.num_stocks = num_stocks
        self.num_features = num_features
        self.pred_features = pred_features
        self.seq_len = seq_len
        self.d_model = d_model

        # Wavelet-based decoupling layer
        self.decouple = DecouplingFlowLayer(d_model=d_model, num_features=num_features,num_heads=num_heads)

        # Dual-frequency spatiotemporal encoder
        self.dual_freq_encoder = DualFrequencyEncoder(seq_len=seq_len, d_model=d_model, num_stocks=num_stocks, num_heads=num_heads)

        # Dual-frequency fusion decoder
        self.dual_freq_fusion_decoder = DualFrequencyFusionDecoder(seq_len, d_model, num_features)


    def forward(self, x, ts) -> dict:
        # x: (batch_size, seq_len, num_stocks, num_features)
        # ts: (batch_size, seq_len) time slot indices
        # Decompose input into low and high frequency components
        X_l, X_h = self.decouple(x)  # Both: (batch_size, seq_len, num_stocks, d_model)
        returns = x[..., self.pred_features[1]]  # (batch_size, seq_len, num_stocks)

        # Encode
        X_l_gat, X_h_gat = self.dual_freq_encoder(X_l, X_h, ts, returns)

        # Decode
        predictions = self.dual_freq_fusion_decoder(X_l_gat, X_h_gat)

        return predictions