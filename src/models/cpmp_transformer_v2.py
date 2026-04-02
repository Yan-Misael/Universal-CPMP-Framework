import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.attention import CrossAttentionBlock
from models.base.transformer import Transformer

class CPMPTransformer(Transformer):
    def __init__(self, d_model=64, nhead=8, num_layers=4, ff_dim_multiplier=4, dropout=0.1, max_height=10):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ff_dim_multiplier=ff_dim_multiplier,
            dropout=dropout
        )
        self.d_model = d_model
        self.nhead = nhead

        # 1. Embeddings de entrada
        self.group_embedding = nn.Linear(1, d_model)
        # Cambio: Usamos nn.Embedding para posiciones discretas (Tiers)
        self.pos_embedding = nn.Embedding(max_height, d_model) 
        self.stack_query_embedding = nn.Parameter(torch.randn(1, 1, d_model))
        self.empty_slot_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 2. Bloques de Atención
        self.state_cross_attn = CrossAttentionBlock(d_model, nhead, ff_dim_multiplier, dropout)
        self.source_refiner = CrossAttentionBlock(d_model, nhead, ff_dim_multiplier, dropout)
        self.target_refiner = CrossAttentionBlock(d_model, nhead, ff_dim_multiplier, dropout)
        
        # 3. Capas de Proyección
        self.source_projector = nn.Linear(d_model, 1)
        self.target_projector = nn.Linear(d_model, 1)

    def forward(self, G, P, I, S, H):
        """
        G: (B, N, 1) - Group values
        P: (B, N, 1) - Tier positions (0 a H-1)
        I: (B, N, 1) - Stack indices
        """
        device = G.device
        S_val = int(S[0].item())
        H_val = int(H[0].item())
        batch_size = G.size(0)

        # --- FASE 1: Construcción de la Rejilla ---
        if G.dim() == 2: G = G.unsqueeze(-1)
        if P.dim() == 2: P = P.unsqueeze(-1)
        if I.dim() == 2: I = I.unsqueeze(-1)

        # Cambio: P ahora se usa como índice para el Embedding
        # Aseguramos que P sea Long y eliminamos la última dimensión para indexar
        pos_idx = P.squeeze(-1).long()
        containers_rep = self.group_embedding(G.float()) + self.pos_embedding(pos_idx)

        grid = self.empty_slot_token.to(containers_rep.dtype).expand(batch_size, S_val, H_val, self.d_model).clone()

        idx_b = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, G.size(1))
        idx_s = I.squeeze(-1).long()
        idx_h = P.squeeze(-1).long()

        grid[idx_b, idx_s, idx_h] = containers_rep
        
        full_kv = grid.view(batch_size, S_val * H_val, self.d_model)

        # --- FASE 2: Máscaras Base ---
        mask_indices = torch.arange(S_val * H_val, device=device).view(1, 1, S_val * H_val)
        s_starts = torch.arange(S_val, device=device).view(1, S_val, 1) * H_val
        s_ends = s_starts + H_val
        base_mask = (mask_indices < s_starts) | (mask_indices >= s_ends)
        base_mask_batched = base_mask.expand(batch_size, -1, -1)

        # --- FASE 3: Representaciones de Estado ---
        stacks_init = self.stack_query_embedding.expand(batch_size, S_val, -1)

        current_stacks = self.state_cross_attn(
            stacks_init, full_kv, full_kv, 
            attn_mask=base_mask_batched.repeat_interleave(self.nhead, dim=0)
        )

        source_mask = base_mask_batched.clone()
        top_global_indices = torch.arange(S_val, device=device) * H_val
        source_mask[:, torch.arange(S_val), top_global_indices] = True
        
        source_states = self.state_cross_attn(
            stacks_init, full_kv, full_kv, 
            attn_mask=source_mask.repeat_interleave(self.nhead, dim=0)
        )

        S_range = torch.arange(S_val, device=device)
        i_idx, j_idx = torch.meshgrid(S_range, S_range, indexing='ij')
        mask_valid = i_idx != j_idx
        indices_i = i_idx[mask_valid] 
        indices_j = j_idx[mask_valid] 
        num_actions = indices_j.size(0)
        
        target_queries = stacks_init[:, indices_j, :]
        target_mask = base_mask_batched[:, indices_j, :].clone()
        
        b_idx = torch.arange(batch_size, device=device).view(-1, 1)
        a_idx = torch.arange(num_actions, device=device).view(1, -1)
        src_slot_idx = (indices_i * H_val).view(1, -1)
        
        target_mask[b_idx, a_idx, src_slot_idx] = False
        
        target_states = self.state_cross_attn(
            target_queries, full_kv, full_kv, 
            attn_mask=target_mask.repeat_interleave(self.nhead, dim=0)
        )

        # --- FASE 4: Proyección de Scores ---
        source_refined = self.source_refiner(source_states, current_stacks, current_stacks)
        target_refined = self.target_refiner(target_states, current_stacks[:, indices_j, :], current_stacks[:, indices_j, :])

        s_scores = self.source_projector(source_refined)
        t_scores = self.target_projector(target_refined)
        
        final_scores = s_scores[:, indices_i, :] + t_scores 
        final_scores = final_scores.squeeze(-1)

        # --- FASE 5: Enmascaramiento ---
        counts = torch.zeros(batch_size, S_val, device=device)
        counts.scatter_add_(1, I.squeeze(-1).long(), torch.ones_like(I.squeeze(-1), dtype=torch.float32))

        is_empty = (counts == 0)
        is_full = (counts == H_val)
        invalid_mask = is_empty[:, indices_i] | is_full[:, indices_j]
        final_scores.masked_fill_(invalid_mask, -1e10)
        
        return final_scores