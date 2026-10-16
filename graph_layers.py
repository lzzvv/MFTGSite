import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from dagatv2_layers import GATv2WithDistanceBias, GATv2WithRBFDistanceBias


def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class Normalize(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        # Reshape
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias


class TransformerLayer(nn.Module):

    def __init__(
        self,
        num_hidden,
        num_in,
        num_heads=8,
        dropout=0.2,
        distance_alpha=0.5,
        use_rbf_distance=True,
        num_rbf=16,
        use_gatv2=True,
        use_gvp=True,
        use_gated_fusion=False
    ):
        super(TransformerLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()
        self.num_hidden = num_hidden
        self.use_gatv2 = use_gatv2
        self.use_gvp = use_gvp
        self.use_rbf_distance = use_rbf_distance
        self.use_gated_fusion = use_gated_fusion

        self.norm = nn.ModuleList([Normalize(num_hidden) for _ in range(4)])

        # Branch 1: DAGATv2
        if self.use_gatv2:
            if use_rbf_distance:
                self.attention = GATv2WithRBFDistanceBias(
                    num_hidden, num_in, num_heads,
                    distance_alpha=distance_alpha,
                    dropout=dropout,
                    num_rbf=num_rbf,
                )
            else:
                self.attention = GATv2WithDistanceBias(
                    num_hidden, num_in, num_heads,
                    distance_alpha=distance_alpha,
                    dropout=dropout,
                )
        else:
            self.attention = None

        # Branch 2: GVP
        if self.use_gvp:
            self.gvp = GVPBranch(
                hidden_dim=num_hidden,
                num_rbf=num_rbf,
                v_dim=8,
                num_layers=2,
                dropout=dropout,
            )
        else:
            self.gvp = None

        # Fusion layer
        if self.use_gatv2 and self.use_gvp:
            if self.use_gated_fusion:
                self.gate_proj = nn.Linear(num_hidden * 2, num_hidden, bias=True)
                nn.init.xavier_uniform_(self.gate_proj.weight)
                nn.init.zeros_(self.gate_proj.bias)
                self.fusion_proj = None
            else:
                self.fusion_proj = nn.Linear(num_hidden * 2, num_hidden, bias=True)
                self.gate_proj = None
        else:
            self.fusion_proj = None
            self.gate_proj = None

        # FFN with 6x expansion
        self.dense1 = PositionWiseFeedForward(num_hidden, num_hidden * 6)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(
        self,
        h_V,
        h_EV,
        adj,
        h0,
        l,
        mask_V=None,
        mask_attend=None,
        distances=None,
        X=None,
        E_idx=None,
    ):
        if self.use_gatv2:
            dh_egat = self.attention(h_V, h_EV, mask_attend, distances=distances)
            h_egat = self.norm[0](h_V + self.dropout(dh_egat))
        else:
            h_egat = h_V

        # Branch 2: GVP (scalar + vector features, E(3)-equivariant)
        if self.use_gvp:
            h_gvp = self.gvp(h_V, X, E_idx, mask_V)
        else:
            h_gvp = None

        # Fusion: concatenation or gated fusion
        if self.use_gatv2 and self.use_gvp:
            if self.use_gated_fusion:
                h_cat = torch.cat([h_egat, h_gvp], dim=-1)  # [B, N, 256]
                gate = torch.sigmoid(self.gate_proj(h_cat))  # [B, N, 128]
                h_fused = gate * h_egat + (1 - gate) * h_gvp  # [B, N, 128]
                h_fused = self.norm[1](h_fused)
                h_fused = self.relu(h_fused)
                h_V = h_fused
            else:
                h_fused = torch.cat([h_egat, h_gvp], dim=-1)  # [B, N, 256]
                h_fused = self.fusion_proj(h_fused)  # [B, N, 128]
                h_V = self.norm[1](h_V + self.dropout(h_fused))
        elif self.use_gatv2:
            h_V = h_egat
        elif self.use_gvp:
            h_V = h_gvp

        dh = self.dense1(h_V)
        h_V = self.norm[2](h_V + self.dropout(dh))

        # Apply mask
        if mask_V is not None:
            mask_V_expanded = mask_V.unsqueeze(-1)
            h_V = mask_V_expanded * h_V

        return h_V


class PositionWiseFeedForward(nn.Module):

    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        # SwiGLU requires three linear projections (no bias)
        self.gate_proj = nn.Linear(num_hidden, num_ff, bias=False)  # Gate pathway
        self.up_proj = nn.Linear(num_hidden, num_ff, bias=False)  # Value pathway
        self.down_proj = nn.Linear(num_ff, num_hidden, bias=False)  # Output projection

    def forward(self, h_V):
        gate = F.silu(self.gate_proj(h_V))
        value = self.up_proj(h_V)
        return self.down_proj(gate * value)



class GVPBranch(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 128,
        num_rbf: int = 16,
        v_dim: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        import gvp_layers as gvp

        self.hidden_dim = hidden_dim
        self.num_rbf = num_rbf
        self.v_dim = v_dim

        # Node embedding: scalar -> (scalar, vector)
        self.W_v = nn.Sequential(
            gvp.LayerNorm((hidden_dim, 0)),
            gvp.GVP(
                (hidden_dim, 0),
                (hidden_dim, v_dim),
                activations=(None, None),
                vector_gate=True,
            ),
        )

        # Edge embedding: (RBF, direction) -> (scalar, vector)
        edge_s_dim = hidden_dim // 4
        edge_v_dim = 1
        self.W_e = nn.Sequential(
            gvp.LayerNorm((num_rbf, 1)),
            gvp.GVP(
                (num_rbf, 1),
                (edge_s_dim, edge_v_dim),
                activations=(None, None),
                vector_gate=True,
            ),
        )

        # Stacked GVP conv layers
        self.layers = nn.ModuleList([
            gvp.GVPConvLayer(
                (hidden_dim, v_dim),
                (edge_s_dim, edge_v_dim),
                n_message=3,
                n_feedforward=2,
                drop_rate=dropout,
                activations=(F.relu, None),
                vector_gate=True,
                residual=True,
            )
            for _ in range(num_layers)
        ])

        # Output layer: (scalar, vector) -> scalar
        self.W_out = nn.Sequential(
            gvp.LayerNorm((hidden_dim, v_dim)),
            gvp.GVP(
                (hidden_dim, v_dim),
                (hidden_dim, 0),
                activations=(F.relu, None),
                vector_gate=True,
            ),
        )

        self.D_min = 0.0
        self.D_max = 20.0
        self.D_count = num_rbf

    def _rbf(self, D):
        device = D.device
        D_mu = torch.linspace(self.D_min, self.D_max, self.D_count, device=device)
        D_sigma = (self.D_max - self.D_min) / self.D_count
        D_expand = D.unsqueeze(-1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF  # [E, num_rbf]

    def _build_edge_index(self, E_idx):
        L, K = E_idx.shape
        node_in = torch.arange(L, device=E_idx.device).view(L, 1).expand(L, K)
        node_out = E_idx
        node_in = node_in.reshape(-1)
        node_out = node_out.reshape(-1)
        edge_index = torch.stack([node_in, node_out], dim=0)
        return edge_index

    def _build_edge_features(self, X, edge_index):
        vectors = X[edge_index[0]] - X[edge_index[1]]  # [E, 3]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [E, 1]
        rbf = self._rbf(lengths.squeeze(-1))  # [E, num_rbf]
        directions = torch.nan_to_num(torch.div(vectors, lengths)).unsqueeze(-2)  # [E, 1, 3]
        h_E = (rbf, directions)
        return h_E

    def forward(self, h_V, X, E_idx, mask):
        B, L, _ = h_V.shape
        device = h_V.device

        # Step 1: build batched data by merging all samples
        all_h_V = []
        all_X = []
        all_edge_index = []
        node_offsets = []
        sample_node_counts = []

        current_offset = 0
        for b in range(B):
            mask_b = mask[b].bool()  # [L]
            h_V_b = h_V[b][mask_b]  # [L_valid, 128]
            X_b = X[b][mask_b]  # [L_valid, 3]
            E_idx_b_raw = E_idx[b][mask_b]  # [L_valid, K]

            L_valid = mask_b.sum().item()
            old_to_new = torch.full((L,), -1, dtype=torch.long, device=device)
            old_to_new[mask_b] = torch.arange(L_valid, device=device)
            E_idx_b = old_to_new[E_idx_b_raw]  # [L_valid, K]

            edge_index_b = self._build_edge_index(E_idx_b)  # [2, E]
            valid_edge_mask = (edge_index_b[1] >= 0)
            edge_index_b = edge_index_b[:, valid_edge_mask]  # [2, E_valid]
            edge_index_b = edge_index_b + current_offset

            all_h_V.append(h_V_b)
            all_X.append(X_b)
            all_edge_index.append(edge_index_b)
            node_offsets.append(current_offset)
            sample_node_counts.append(L_valid)

            current_offset += L_valid

        all_h_V = torch.cat(all_h_V, dim=0)  # [N_total, 128]
        all_X = torch.cat(all_X, dim=0)  # [N_total, 3]
        all_edge_index = torch.cat(all_edge_index, dim=1)  # [2, E_total]

        # Step 2: batched GVP forward pass over all nodes at once
        h_E = self._build_edge_features(all_X, all_edge_index)  # (s_e, V_e)
        h_V_gvp = self.W_v(all_h_V)  # (s, V)
        h_E = self.W_e(h_E)

        for layer in self.layers:
            h_V_gvp = layer(h_V_gvp, all_edge_index, h_E)

        out_all = self.W_out(h_V_gvp)  # [N_total, 128]

        # Step 3: split back into the original [B, L, 128] batch layout
        outputs = []
        for b in range(B):
            offset = node_offsets[b]
            count = sample_node_counts[b]
            out_b = out_all[offset:offset + count]  # [L_valid, 128]

            out_padded = torch.zeros(L, self.hidden_dim, device=device)
            mask_b = mask[b].bool()
            out_padded[mask_b] = out_b
            outputs.append(out_padded)

        return torch.stack(outputs, dim=0)  # [B, L, 128]
