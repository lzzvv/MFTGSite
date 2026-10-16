import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2WithDistanceBias(nn.Module):

    def __init__(
        self, num_hidden, num_in, num_heads=4, distance_alpha=0.5, dropout=0.1
    ):
        super(GATv2WithDistanceBias, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.distance_alpha = distance_alpha
        self.dropout = dropout

        # Per-head dimension
        assert num_hidden % num_heads == 0, "num_hidden must be divisible by num_heads"
        self.d = num_hidden // num_heads

        # Linear transformations
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_in, num_hidden, bias=False)
        self.W_V = nn.Linear(num_in, num_hidden, bias=False)
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

        # GATv2 attention mechanism: a^T LeakyReLU(W_l h_i + W_r h_j)
        self.att = nn.Parameter(torch.Tensor(num_heads, self.d))

        # LeakyReLU with negative slope 0.2 (standard for GATv2)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.W_O.weight)
        nn.init.xavier_uniform_(self.att)

    def _masked_softmax(self, attend_logits, mask_attend, dim=-1):
        negative_inf = np.finfo(np.float32).min
        attend_logits = torch.where(
            mask_attend > 0,
            attend_logits,
            torch.tensor(negative_inf).to(attend_logits.device),
        )
        attend = F.softmax(attend_logits, dim)
        attend = mask_attend * attend
        return attend

    def forward(self, h_V, h_E, mask_attend=None, distances=None):
        n_batch, n_nodes, n_neighbors = h_E.shape[:3]
        n_heads = self.num_heads
        d = self.d

        # Step 1: Linear transformations
        Q = self.W_Q(h_V).view(n_batch, n_nodes, n_heads, d)
        K = self.W_K(h_E).view(n_batch, n_nodes, n_neighbors, n_heads, d)
        V = self.W_V(h_E).view(n_batch, n_nodes, n_neighbors, n_heads, d)

        # Step 2: GATv2 attention computation
        # Expand Q: [B, N, H, d] -> [B, N, K, H, d]
        Q_expanded = Q.unsqueeze(2).expand(-1, -1, n_neighbors, -1, -1)

        # GATv2: Use addition (more efficient than concatenation)
        QK_sum = Q_expanded + K  # [B, N, K, H, d]

        # Apply LeakyReLU (key difference from GAT)
        QK_activated = self.leaky_relu(QK_sum)

        # Compute attention logits: a^T * activated_features
        attend_logits = torch.einsum("bnkhd,hd->bnkh", QK_activated, self.att)

        # Transpose to [B, N, H, K]
        attend_logits = attend_logits.transpose(-2, -1)

        # Step 3: Add distance bias (CORE INNOVATION - PRESERVED!)
        if distances is not None and self.distance_alpha > 0:
            distance_bias = -self.distance_alpha * distances.unsqueeze(2).expand(
                -1, -1, n_heads, -1
            )
            attend_logits = attend_logits + distance_bias

        # Step 4: Masked softmax
        if mask_attend is not None:
            mask = mask_attend.unsqueeze(2).expand(-1, -1, n_heads, -1)
            attend = self._masked_softmax(attend_logits, mask)
        else:
            attend = F.softmax(attend_logits, dim=-1)

        # Step 5: Attention dropout (NEW - from official implementation)
        attend = F.dropout(attend, p=self.dropout, training=self.training)

        # Step 6: Aggregate values
        V = V.transpose(2, 3)  # [B, N, H, K, d]
        h_V_update = torch.einsum("bnhk,bnhkd->bnhd", attend, V)
        h_V_update = h_V_update.reshape(n_batch, n_nodes, self.num_hidden)

        # Step 7: Output projection
        h_V_update = self.W_O(h_V_update)

        return h_V_update


class GATv2WithRBFDistanceBias(nn.Module):

    def __init__(
        self,
        num_hidden,
        num_in,
        num_heads=4,
        distance_alpha=0.5,
        dropout=0.1,
        num_rbf=16,
    ):
        super(GATv2WithRBFDistanceBias, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.distance_alpha = distance_alpha  # Keep for reference
        self.dropout = dropout
        self.num_rbf = num_rbf

        # Per-head dimension
        assert num_hidden % num_heads == 0, "num_hidden must be divisible by num_heads"
        self.d = num_hidden // num_heads

        # Linear transformations (same as GATv2WithDistanceBias)
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_in, num_hidden, bias=False)
        self.W_V = nn.Linear(num_in, num_hidden, bias=False)
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

        # GATv2 attention mechanism: a^T LeakyReLU(W_l h_i + W_r h_j)
        self.att = nn.Parameter(torch.Tensor(num_heads, self.d))

        # LeakyReLU with negative slope 0.2 (standard for GATv2)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # RBF centers: uniformly distributed from 0 to 20 Angstroms
        # These are fixed (not learnable) to provide a stable basis
        self.register_buffer("rbf_centers", torch.linspace(0, 20, num_rbf))

        # RBF width (gamma): learnable parameter
        # Initialize to 0.1 for wide distance sensitivity (best performing config)
        # gamma=0.1 -> sigma=2.24Å, captures wide distance range
        self.rbf_gamma = nn.Parameter(torch.ones(1) * 0.1)

        # Projects RBF features to attention bias for each head
        # Architecture: num_rbf -> num_rbf*2 -> num_heads
        self.distance_mlp = nn.Sequential(
            nn.Linear(num_rbf, num_rbf * 2),
            nn.ReLU(),
            nn.Linear(num_rbf * 2, num_heads),
        )

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.W_O.weight)
        nn.init.xavier_uniform_(self.att)

        for layer in self.distance_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def rbf_encoding(self, distances):
        D = distances.unsqueeze(-1)
        centers = self.rbf_centers.view(1, 1, 1, -1)
        gamma = torch.clamp(self.rbf_gamma, min=0.01, max=10.0)
        diff_sq = (D - centers) ** 2
        exponent = torch.clamp(-gamma * diff_sq, min=-10, max=0)
        return torch.exp(exponent)  # [B, N, K, num_rbf]

    def _masked_softmax(self, attend_logits, mask_attend, dim=-1):
        negative_inf = np.finfo(np.float32).min
        attend_logits = torch.where(
            mask_attend > 0,
            attend_logits,
            torch.tensor(negative_inf).to(attend_logits.device),
        )
        attend = F.softmax(attend_logits, dim)
        attend = mask_attend * attend
        return attend

    def forward(self, h_V, h_E, mask_attend=None, distances=None):
        n_batch, n_nodes, n_neighbors = h_E.shape[:3]
        n_heads = self.num_heads
        d = self.d

        # Step 1: Linear transformations
        Q = self.W_Q(h_V).view(n_batch, n_nodes, n_heads, d)
        K = self.W_K(h_E).view(n_batch, n_nodes, n_neighbors, n_heads, d)
        V = self.W_V(h_E).view(n_batch, n_nodes, n_neighbors, n_heads, d)

        # Step 2: GATv2 attention computation
        # Expand Q: [B, N, H, d] -> [B, N, K, H, d]
        Q_expanded = Q.unsqueeze(2).expand(-1, -1, n_neighbors, -1, -1)

        # GATv2: Use addition (more efficient than concatenation)
        QK_sum = Q_expanded + K  # [B, N, K, H, d]

        # Apply LeakyReLU (key difference from GAT)
        QK_activated = self.leaky_relu(QK_sum)

        # Compute attention logits: a^T * activated_features
        attend_logits = torch.einsum("bnkhd,hd->bnkh", QK_activated, self.att)

        # Transpose to [B, N, H, K]
        attend_logits = attend_logits.transpose(-2, -1)

        if distances is not None:
            distance_bias_fixed = -self.distance_alpha * distances.unsqueeze(2)  # [B, N, 1, K]
            rbf_features = self.rbf_encoding(distances)                          # [B, N, K, num_rbf]
            distance_bias_rbf = self.distance_mlp(rbf_features).transpose(-2, -1) * 0.5  # [B, N, H, K]
            attend_logits = attend_logits + distance_bias_fixed + distance_bias_rbf

        # Step 4: Masked softmax
        if mask_attend is not None:
            mask = mask_attend.unsqueeze(2).expand(-1, -1, n_heads, -1)
            attend = self._masked_softmax(attend_logits, mask)
        else:
            attend = F.softmax(attend_logits, dim=-1)

        # Step 5: Attention dropout
        attend = F.dropout(attend, p=self.dropout, training=self.training)

        # Step 6: Aggregate values
        V = V.transpose(2, 3)  # [B, N, H, K, d]
        h_V_update = torch.einsum("bnhk,bnhkd->bnhd", attend, V)
        h_V_update = h_V_update.reshape(n_batch, n_nodes, self.num_hidden)

        # Step 7: Output projection
        h_V_update = self.W_O(h_V_update)

        return h_V_update

