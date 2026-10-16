import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):

    def __init__(
        self,
        dim_esm2=256,
        dim_af2=200,
        dim_dssp=64,
        dim_blosum62=64,
        hidden_dim=128,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # ====================================================================
        self.proj_esm2 = nn.Linear(dim_esm2, hidden_dim)
        self.proj_af2 = nn.Linear(dim_af2, hidden_dim)
        self.proj_dssp = nn.Linear(dim_dssp, hidden_dim)
        self.proj_blosum62 = nn.Linear(dim_blosum62, hidden_dim)

        self.norm_esm2 = nn.LayerNorm(hidden_dim)
        self.norm_af2 = nn.LayerNorm(hidden_dim)
        self.norm_dssp = nn.LayerNorm(hidden_dim)
        self.norm_blosum62 = nn.LayerNorm(hidden_dim)
        # ====================================================================
        self.cross_attn_esm2_to_af2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_af2_to_esm2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_attn_esm2_to_dssp = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_dssp_to_esm2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_attn_esm2_to_blosum62 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_blosum62_to_esm2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_attn_af2_to_dssp = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_dssp_to_af2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_attn_af2_to_blosum62 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_blosum62_to_af2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_attn_dssp_to_blosum62 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_blosum62_to_dssp = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ====================================================================
        # LayerNorm and Dropout
        # ====================================================================
        self.norm_esm2_enhanced = nn.LayerNorm(hidden_dim)
        self.norm_af2_enhanced = nn.LayerNorm(hidden_dim)
        self.norm_dssp_enhanced = nn.LayerNorm(hidden_dim)
        self.norm_blosum62_enhanced = nn.LayerNorm(hidden_dim)

        # ====================================================================
        # FFN (Feed-Forward Network) for each modality
        # ====================================================================
        self.ffn_esm2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ffn_af2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ffn_dssp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ffn_blosum62 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Final LayerNorm for FFN outputs
        self.norm_ffn_esm2 = nn.LayerNorm(hidden_dim)
        self.norm_ffn_af2 = nn.LayerNorm(hidden_dim)
        self.norm_ffn_dssp = nn.LayerNorm(hidden_dim)
        self.norm_ffn_blosum62 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # 128 (ESM-2) + 128 (AF2) + 128 (DSSP) + 128 (BLOSUM62) = 512
        fusion_input_dim = hidden_dim * 4  # 512
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),  # 512 → 256
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),  # 256 → 128
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, esm2, af2, dssp, blosum62, mask=None):
        
        # ====================================================================
        # Step 1
        # ====================================================================
        h_esm2 = self.norm_esm2(self.proj_esm2(esm2))  # [B, L, 128]
        h_af2 = self.norm_af2(self.proj_af2(af2))  # [B, L, 128]
        h_dssp = self.norm_dssp(self.proj_dssp(dssp))  # [B, L, 128]
        h_blosum62 = self.norm_blosum62(self.proj_blosum62(blosum62))  # [B, L, 128]

        # ====================================================================
        # Step 2
        # ====================================================================
        key_padding_mask = None
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.squeeze(-1)  # [B, L, 1] → [B, L]
            key_padding_mask = ~mask.bool()  # True = padding, False = valid
        # ====================================================================
        esm2_from_af2, _ = self.cross_attn_esm2_to_af2(
            query=h_esm2,
            key=h_af2,
            value=h_af2,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        af2_from_esm2, _ = self.cross_attn_af2_to_esm2(
            query=h_af2,
            key=h_esm2,
            value=h_esm2,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        esm2_from_dssp, _ = self.cross_attn_esm2_to_dssp(
            query=h_esm2,
            key=h_dssp,
            value=h_dssp,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        dssp_from_esm2, _ = self.cross_attn_dssp_to_esm2(
            query=h_dssp,
            key=h_esm2,
            value=h_esm2,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        esm2_from_blosum62, _ = self.cross_attn_esm2_to_blosum62(
            query=h_esm2,
            key=h_blosum62,
            value=h_blosum62,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        blosum62_from_esm2, _ = self.cross_attn_blosum62_to_esm2(
            query=h_blosum62,
            key=h_esm2,
            value=h_esm2,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        af2_from_dssp, _ = self.cross_attn_af2_to_dssp(
            query=h_af2,
            key=h_dssp,
            value=h_dssp,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        dssp_from_af2, _ = self.cross_attn_dssp_to_af2(
            query=h_dssp,
            key=h_af2,
            value=h_af2,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        af2_from_blosum62, _ = self.cross_attn_af2_to_blosum62(
            query=h_af2,
            key=h_blosum62,
            value=h_blosum62,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        blosum62_from_af2, _ = self.cross_attn_blosum62_to_af2(
            query=h_blosum62,
            key=h_af2,
            value=h_af2,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        dssp_from_blosum62, _ = self.cross_attn_dssp_to_blosum62(
            query=h_dssp,
            key=h_blosum62,
            value=h_blosum62,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        blosum62_from_dssp, _ = self.cross_attn_blosum62_to_dssp(
            query=h_blosum62,
            key=h_dssp,
            value=h_dssp,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        # ====================================================================
        # Step 4
        # ====================================================================
        esm2_attn_out = self.norm_esm2_enhanced(
            h_esm2 + self.dropout((esm2_from_af2 + esm2_from_dssp + esm2_from_blosum62) / 3.0)
        )
        esm2_enhanced = self.norm_ffn_esm2(esm2_attn_out + self.dropout(self.ffn_esm2(esm2_attn_out)))

        af2_attn_out = self.norm_af2_enhanced(
            h_af2 + self.dropout((af2_from_esm2 + af2_from_dssp + af2_from_blosum62) / 3.0)
        )
        af2_enhanced = self.norm_ffn_af2(af2_attn_out + self.dropout(self.ffn_af2(af2_attn_out)))

        dssp_attn_out = self.norm_dssp_enhanced(
            h_dssp + self.dropout((dssp_from_esm2 + dssp_from_af2 + dssp_from_blosum62) / 3.0)
        )
        dssp_enhanced = self.norm_ffn_dssp(dssp_attn_out + self.dropout(self.ffn_dssp(dssp_attn_out)))

        blosum62_attn_out = self.norm_blosum62_enhanced(
            h_blosum62 + self.dropout((blosum62_from_esm2 + blosum62_from_af2 + blosum62_from_dssp) / 3.0)
        )
        blosum62_enhanced = self.norm_ffn_blosum62(blosum62_attn_out + self.dropout(self.ffn_blosum62(blosum62_attn_out)))

        # ====================================================================

        fused = torch.cat(
            [esm2_enhanced, af2_enhanced, dssp_enhanced, blosum62_enhanced], dim=-1
        )  # [B, L, hidden_dim*4] = [B, L, 512]
        fused = self.fusion(fused)  # [B, L, hidden_dim] = [B, L, 128]

        return fused


class GatedBranchFusion(nn.Module):

    def __init__(self, hidden_dim=128):
        super().__init__()

        self.gate = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.sigmoid = nn.Sigmoid()

        self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_egat, h_gcnii):

        h_concat = torch.cat([h_egat, h_gcnii], dim=-1)  # [B, L, hidden_dim*2]

        gate = self.sigmoid(self.gate(h_concat))  # [B, L, hidden_dim*2]

        h_gated = gate * h_concat  # [B, L, hidden_dim*2]

        h_fused = self.fusion_proj(h_gated)  # [B, L, hidden_dim]
        h_fused = self.norm(h_fused)

        return h_fused


class DenseConnection(nn.Module):

    def __init__(self, hidden_dim=128, num_layers=5):
        super().__init__()

        self.dense_proj = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim * 2, bias=True),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, h_V_all):
        h_V_dense = torch.cat(h_V_all, dim=-1)  # [B, L, hidden_dim*num_layers]
        h_V_dense = self.dense_proj(h_V_dense)  # [B, L, hidden_dim*2]
        return h_V_dense


if __name__ == "__main__":
    print("Testing CrossModalAttention...")
    B, L = 2, 100
    esm2 = torch.randn(B, L, 256)
    af2 = torch.randn(B, L, 128)
    dssp = torch.randn(B, L, 128)
    blosum62 = torch.randn(B, L, 128)
    mask = torch.ones(B, L)

    cross_modal = CrossModalAttention(
        dim_esm2=256, dim_af2=128, dim_dssp=128, dim_blosum62=128,
        hidden_dim=128, num_heads=4,
    )
    fused = cross_modal(esm2, af2, dssp, blosum62, mask)
    print(f"  Input: ESM-2 {esm2.shape}, AF2 {af2.shape}, DSSP {dssp.shape}, BLOSUM62 {blosum62.shape}")
    print(f"  Output: {fused.shape}")
    print(f"  Parameters: {sum(p.numel() for p in cross_modal.parameters()):,}")
    print()

    print("Testing GatedBranchFusion...")
    h_egat = torch.randn(B, L, 128)
    h_gcnii = torch.randn(B, L, 128)

    gated_fusion = GatedBranchFusion(hidden_dim=128)
    h_fused = gated_fusion(h_egat, h_gcnii)
    print(f"  Input: DAGATv2 {h_egat.shape}, GCNII {h_gcnii.shape}")
    print(f"  Output: {h_fused.shape}")
    print(f"  Parameters: {sum(p.numel() for p in gated_fusion.parameters()):,}")
    print()

    print("Testing DenseConnection...")
    h_V_all = [torch.randn(B, L, 128) for _ in range(5)]

    dense_conn = DenseConnection(hidden_dim=128, num_layers=5)
    h_V_dense = dense_conn(h_V_all)
    print(f"  Input: 5 layers of {h_V_all[0].shape}")
    print(f"  Output: {h_V_dense.shape}")
    print(f"  Parameters: {sum(p.numel() for p in dense_conn.parameters()):,}")
    print()

    print("All tests passed.")
