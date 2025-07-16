import torch
import pywt
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
import numpy as np
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
        X_l = X_l_up.reshape(batch_size, num_stocks, seq_len).permute(0, 2, 1)  # (batch_size, seq_len, num_stocks)
        X_h = X_h_up.reshape(batch_size, num_stocks, seq_len).permute(0, 2, 1)  # (batch_size, seq_len, num_stocks)

        # Replace return feature in original input
        X_l_full = x.clone()
        X_l_full[:, :, :, self.return_index] = X_l  # (batch_size, seq_len, num_stocks, num_features)
        X_h_full = x.clone()
        X_h_full[:, :, :, self.return_index] = X_h  # (batch_size, seq_len, num_stocks, num_features)

        # Project to d_model dimensions
        X_l_proj = self.Wg(X_l_full.reshape(-1, num_features)).reshape(batch_size, seq_len, num_stocks, self.d_model)
        X_h_proj = self.Wh(X_h_full.reshape(-1, num_features)).reshape(batch_size, seq_len, num_stocks, self.d_model)

        return X_l_proj, X_h_proj  # Both: (batch_size, seq_len, num_stocks, d_model)

class GraphAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super(GraphAttentionLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x):
        # x: (batch_size, seq_len, num_stocks, d_model)
        batch_size, seq_len, num_stocks, d_model = x.shape
        
        # Flatten for multi-head attention
        x_flat = x.reshape(batch_size * seq_len, num_stocks, d_model)
        
        Q = self.query(x_flat).reshape(batch_size * seq_len, num_stocks, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x_flat).reshape(batch_size * seq_len, num_stocks, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x_flat).reshape(batch_size * seq_len, num_stocks, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size * seq_len, num_stocks, d_model)
        output = self.out(attn_output)
        
        # Reshape back
        output = output.reshape(batch_size, seq_len, num_stocks, d_model)
        
        # Residual connection and layer norm
        output = self.ln(output + x)
        
        # Feed forward
        output = self.ff(output) + output
        
        return output
class DualFrequencyEncoder(nn.Module):
    def __init__(self, seq_len, d_model, num_stocks, num_heads, kernel_size=2):
        super(DualFrequencyEncoder, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_stocks = num_stocks

        # Temporal attention for low-frequency components
        self.temporal_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        # Dilated convolution with padding to preserve sequence length
        self.dilated_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, dilation=2, padding=1)
        self.relu = nn.ReLU()

        # Positional encoding
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', pe.unsqueeze(0))

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
        ts_index = ts_index.long()  # Ensure integer type for one_hot
        one_hot = F.one_hot(ts_index, num_classes=self.num_time_slots).float()  # (batch, seq, 252)
        time_embed = self.time_embedding(one_hot)  # (batch, seq, d_model)
        return time_embed  # (batch, seq, d_model)

    def embed_spatial(self, returns):
        # returns: (batch_size, seq_len, num_stocks)
        # Compute Spearman correlation matrix and use as spatial embedding
        batch_size, seq_len, num_stocks = returns.shape
        
        # Compute correlation for each sample in the batch
        spatial_embeddings = []
        for b in range(batch_size):
            corr_matrix = self.compute_spearman_correlation(returns[b])  # (num_stocks, num_stocks)
            # Convert correlation matrix to spatial embedding
            spatial_emb = torch.matmul(corr_matrix, self.spatial_embedding)  # (num_stocks, d_model)
            spatial_embeddings.append(spatial_emb)
        
        spatial_embeddings = torch.stack(spatial_embeddings, dim=0)  # (batch_size, num_stocks, d_model)
        spatial_embeddings = spatial_embeddings.unsqueeze(1).repeat(1, seq_len, 1, 1)  # (batch_size, seq_len, num_stocks, d_model)
        
        return spatial_embeddings
        
    def compute_spearman_correlation(self, returns):
        # returns: (seq_len, num_stocks)
        seq_len, num_stocks = returns.shape
        
        # Convert to numpy for spearman correlation calculation
        returns_np = returns.detach().cpu().numpy()
        
        # Compute spearman correlation matrix
        try:
            corr_matrix = np.zeros((num_stocks, num_stocks))
            for i in range(num_stocks):
                for j in range(num_stocks):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        corr, _ = spearmanr(returns_np[:, i], returns_np[:, j])
                        corr_matrix[i, j] = corr if not np.isnan(corr) else 0.0
        except:
            # Fallback to identity matrix if correlation computation fails
            print("Spearman correlation computation failed, using identity matrix.")
            corr_matrix = np.eye(num_stocks)
        
        return torch.tensor(corr_matrix, dtype=returns.dtype, device=returns.device)
       

    def forward(self, X_l, X_h, ts_index, returns):
        # X_l, X_h: (batch_size, seq_len, num_stocks, d_model)
        # ts_index: (batch_size, seq_len)
        # returns: (batch_size, seq_len, num_stocks)
        batch_size, seq_len, num_stocks, d_model = X_l.shape

        # Temporal Attention on low-frequency
        X_l_flat = X_l.reshape(batch_size * num_stocks, seq_len, d_model)
        X_tatt, _ = self.temporal_attention(X_l_flat, X_l_flat, X_l_flat)  # (batch_size*num_stocks, seq_len, d_model)
        X_tatt = X_tatt.reshape(batch_size, seq_len, num_stocks, d_model)

        # Dilated Conv on high-frequency
        X_h_perm = X_h.permute(0, 2, 3, 1)  # (batch_size, num_stocks, d_model, seq_len)
        X_h_flat = X_h_perm.reshape(batch_size * num_stocks, d_model, seq_len)
        X_conv = self.dilated_conv(X_h_flat)  # (batch_size*num_stocks, d_model, seq_len)
        X_conv = self.relu(X_conv)
        X_conv = X_conv.reshape(batch_size, num_stocks, d_model, seq_len).permute(0, 3, 1, 2)  # (batch_size, seq_len, num_stocks, d_model)
        
        # Temporal Graph Embedding
        p_tem = self.embed_time_slots(ts_index)  # (batch, seq, d_model)
        p_tem = p_tem.unsqueeze(2).repeat(1, 1, num_stocks, 1)  # (batch, seq, num_stocks, d_model)

        # Spatial Embedding
        p_spa = self.embed_spatial(returns)  # (batch, seq, num_stocks, d_model)
        
        # Combine embeddings
        X_l = X_tatt + p_tem + p_spa
        X_h = X_conv + p_tem + p_spa

        # Self-Attention (simplified, should be GAT)
        X_l_gat = self.gat_l(X_l)  # (batch, seq, num_stocks, d_model)
        X_h_gat = self.gat_h(X_h)  # (batch, seq, num_stocks, d_model)

        return X_l_gat, X_h_gat

class DualFrequencyFusionDecoder(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, num_heads):
        super(DualFrequencyFusionDecoder, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        # Predictors
        self.predictor_l = nn.Conv2d(self.seq_len, self.pred_len, (1,1))
        self.predictor_h = nn.Conv2d(self.seq_len, self.pred_len, (1,1))

        # Positional encoding
        position = torch.arange(pred_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(pred_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', pe.unsqueeze(0))

        # Attention layers
        self.attn_self = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.attn_cross = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)

        # Output layers
        self.fc_reg= nn.Linear(d_model, 1)
        self.fc_cla = nn.Linear(d_model, 1)

    def forward(self, X_l_gat, X_h_gat) -> dict:
        # X_l_gat, X_h_gat: (batch_size, seq_len, num_stocks, d_model)
        batch_size, pred_len, num_stocks, d_model = X_l_gat.shape

        # Predictors
        pred_l = self.predictor_l(X_l_gat)  # (batch_size, pred_len, num_stocks, d_model)
        pred_h = self.predictor_h(X_h_gat)  # (batch_size, pred_len, num_stocks, d_model)

        batch_size, pred_len, num_stocks, d_model = pred_l.shape

        pred_l = pred_l.reshape(batch_size * num_stocks, pred_len, d_model)
        pred_h = pred_h.reshape(batch_size * num_stocks, pred_len, d_model)

        # Add positional encoding - remove the slicing since positional_encoding already has pred_len dimensions
        input_l = pred_l + self.positional_encoding  # (batch_size*num_stocks, pred_len, d_model)
        input_h = pred_h + self.positional_encoding  # (batch_size*num_stocks, pred_len, d_model)

        # Fusion Attention
        attn_self_out, _ = self.attn_self(input_l, input_l, input_l)
        attn_cross_out, _ = self.attn_cross(input_l, input_h, input_h)
        output = attn_self_out + attn_cross_out  # (batch_size*num_stocks, pred_len, d_model)

        # Low-frequency outputs
        Y_l_reg = self.fc_reg(pred_l).reshape(batch_size, pred_len, num_stocks)  # (batch_size, pred_len, num_stocks)
        Y_l_cla = self.fc_cla(pred_l).reshape(batch_size, pred_len, num_stocks)  # (batch_size, pred_len, num_stocks)
        # Fused outputs
        Y_reg = self.fc_reg(output).reshape(batch_size, pred_len, num_stocks)  # (batch_size, pred_len, num_stocks)
        Y_cla = self.fc_cla(output).reshape(batch_size, pred_len, num_stocks)  # (batch_size, pred_len, num_stocks)

        return {
            "lreg": Y_l_reg,
            "lcla": Y_l_cla,
            "reg": Y_reg,
            "cla": Y_cla,
        }

class StockFormer(nn.Module):
    """
    StockFormer model for time series forecasting.
    """
    def __init__(self, num_stocks, seq_len=20, pred_len=2, num_features=362, d_model=128, num_heads=1, pred_features=[0, 1]):
        super(StockFormer, self).__init__()
        self.num_stocks = num_stocks
        self.num_features = num_features
        self.pred_features = pred_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        # Dropout layer
        self.dropout = nn.Dropout(p=0.2)

        # Wavelet-based decoupling layer
        self.decouple = DecouplingFlowLayer(d_model=d_model, num_features=num_features)

        # Dual-frequency spatiotemporal encoder
        self.dual_freq_encoder = DualFrequencyEncoder(seq_len=seq_len, d_model=d_model, num_stocks=num_stocks, num_heads=num_heads)

        # Dual-frequency fusion decoder
        self.dual_freq_fusion_decoder = DualFrequencyFusionDecoder(seq_len=seq_len, pred_len=pred_len, d_model=d_model, num_heads=num_heads)

    def forward(self, x, ts) -> dict:
        # x: (batch_size, seq_len, num_stocks, num_features)
        # ts: (batch_size, seq_len) time slot indices
        
        # Decompose input into low and high frequency components
        X_l, X_h = self.decouple(x)  # Both: (batch_size, seq_len, num_stocks, d_model)
        
        # Apply dropout to the decomposed features
        X_l = self.dropout(X_l)
        X_h = self.dropout(X_h)

        returns = x[..., self.pred_features[0]]  # (batch_size, seq_len, num_stocks)

        # Encode
        X_l_gat, X_h_gat = self.dual_freq_encoder(X_l, X_h, ts, returns)

        # Apply dropout to encoded features
        X_l_gat = self.dropout(X_l_gat)
        X_h_gat = self.dropout(X_h_gat)

        # Decode
        predictions = self.dual_freq_fusion_decoder(X_l_gat, X_h_gat)

        return predictions