from graph_layers import *
from graph_edges import EdgeFeatures
from cross_modal_attention import CrossModalAttention, DenseConnection
import torch.nn.functional as F

class MODEL(nn.Module):
    def __init__(
        self,
        node_features,
        edge_features,
        hidden_dim,
        num_encoder_layers=4,
        k_neighbors=30,
        augment_eps=0.0,
        dropout=0.2,
        distance_alpha=0.5,
        use_gatv2=True,
        use_rbf_distance=True,
        num_rbf=16,
        use_cross_modal=True,
        use_dense_connection=False,
        use_gvp=False,
        use_gated_fusion=False
    ):
        super(MODEL, self).__init__()

        self.augment_eps = augment_eps
        self.distance_alpha = distance_alpha
        self.use_gatv2 = use_gatv2
        self.use_gvp = use_gvp
        self.use_rbf_distance = use_rbf_distance
        self.num_rbf = num_rbf
        self.use_cross_modal = use_cross_modal
        self.use_dense_connection = use_dense_connection
        self.num_encoder_layers = num_encoder_layers
        self.use_gated_fusion = use_gated_fusion

        # Edge featurization
        self.EdgeFeatures = EdgeFeatures(
            edge_features, top_k=k_neighbors, augment_eps=augment_eps
        )
        self.dropout = nn.Dropout(dropout)

        # Modality-specific projection layers
        self.lin_esm2 = nn.Linear(1280, 256, bias=False)    
        self.lin_af2 = nn.Linear(256, 200, bias=False)      
        self.lin_dssp = nn.Linear(14, 64, bias=False)      
        self.lin_blosum62 = nn.Linear(20, 64, bias=False)   

        # Feature fusion: Cross-Modal Attention or Simple Concatenation
        if use_cross_modal:
            self.cross_modal_fusion = CrossModalAttention(
                dim_esm2=256,
                dim_af2=200,
                dim_dssp=64,
                dim_blosum62=64,
                hidden_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
            )
        else:
            # 584 = 256 (ESM-2) + 200 (AF2) + 64 (DSSP) + 64 (BLOSUM62)
            self.lin_fusion = nn.Linear(584, hidden_dim, bias=True)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)

        self.activation = nn.GELU()

        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                TransformerLayer(
                    hidden_dim,
                    hidden_dim * 2,
                    dropout=dropout,
                    distance_alpha=distance_alpha,
                    use_gatv2=use_gatv2,
                    use_rbf_distance=use_rbf_distance,
                    num_rbf=num_rbf,
                    use_gvp=use_gvp,
                    use_gated_fusion=use_gated_fusion
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # Dense Connection (optional)
        if use_dense_connection:
            self.dense_conn = DenseConnection(
                hidden_dim=hidden_dim,
                num_layers=num_encoder_layers + 1,
            )

        # Output MLP: hidden_dim*2 → 128 → 64 → 1
        self.W_out = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        self.W_out1 = nn.Linear(hidden_dim * 2, 128, bias=True)
        self.norm_out1 = nn.LayerNorm(128)
        self.W_out2 = nn.Linear(128, 64, bias=True)
        self.norm_out2 = nn.LayerNorm(64)
        self.W_out3 = nn.Linear(64, 1, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, V, mask, adj):
        if self.training and self.augment_eps > 0:
            V = V + self.augment_eps * torch.randn_like(V)

        # Split features: ESM-2(1280) + AF2(256) + DSSP(14) + BLOSUM62(20) = 1570
        V_esm2 = V[:, :, :1280]
        V_af2 = V[:, :, 1280:1536]
        V_dssp = V[:, :, 1536:1550]
        V_blosum62 = V[:, :, 1550:1570]

        V_esm2 = self.activation(self.lin_esm2(V_esm2))
        V_af2 = self.activation(self.lin_af2(V_af2))
        V_dssp = self.activation(self.lin_dssp(V_dssp))
        V_blosum62 = self.activation(self.lin_blosum62(V_blosum62))

        if self.use_cross_modal:
            h_V = self.cross_modal_fusion(V_esm2, V_af2, V_dssp, V_blosum62, mask)
        else:
            V = torch.cat([V_esm2, V_af2, V_dssp, V_blosum62], dim=-1)
            h_V = self.activation(self.lin_fusion(V))

        ho = h_V

        if self.use_dense_connection:
            h_V_all = [h_V]

        for i, layer in enumerate(self.encoder_layers):
            if i == 0:
                E, E_idx, D_neighbors = self.EdgeFeatures(X, mask)

            h_E = self.W_e(E)
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
            mask_attend = mask.unsqueeze(-1) * mask_attend

            h_V = layer(
                h_V, h_EV, adj, ho, i + 1,
                mask_V=mask,
                mask_attend=mask_attend,
                distances=D_neighbors,
                X=X,
                E_idx=E_idx,
            )

            if self.use_dense_connection:
                h_V_all.append(h_V)

        if self.use_dense_connection:
            h_V_final = self.dense_conn(h_V_all)
        else:
            h_V_final = torch.cat([h_V, ho], dim=-1)

        logits = self.dropout(self.activation(self.norm_out1(self.W_out1(h_V_final))))
        logits = self.dropout(self.activation(self.norm_out2(self.W_out2(logits))))
        logits = self.W_out3(logits).squeeze(-1)

        return logits
