"""
CascadeEWS Model Architecture
==============================
Advanced Q1-level components:
  1. GraphAttentionNetwork (GAT)     — spatial message passing with attention
  2. GRU temporal encoder            — captures temporal dynamics
  3. Physics-informed constraints    — AR1/variance penalty terms
  4. Uncertainty estimation          — Monte Carlo dropout
  5. EarlyWarning module             — CSD-based risk detection
  6. CascadeSimulator                — SIR-model failure propagation
  7. Explainability hooks            — attention map extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────
class GraphAttentionLayer(nn.Module):
    """
    Multi-head GAT layer.
    Velickovic et al. (2018) — Graph Attention Networks — ICLR
    """
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.2):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = out_dim // n_heads
        self.W        = nn.Linear(in_dim, out_dim, bias=False)
        self.a        = nn.Parameter(torch.zeros(n_heads, 2 * self.head_dim))
        self.dropout  = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(out_dim)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

    def forward(self, x, edge_index, edge_weight=None):
        """
        x          : (N, in_dim)
        edge_index : (2, E)
        edge_weight: (E,) optional
        Returns    : (N, out_dim), attention_weights (E, n_heads)
        """
        N   = x.size(0)
        Wh  = self.W(x).view(N, self.n_heads, self.head_dim)  # (N, H, D)
        src, dst = edge_index[0], edge_index[1]

        Wh_src = Wh[src]  # (E, H, D)
        Wh_dst = Wh[dst]  # (E, H, D)

        # Attention coefficients
        e = torch.cat([Wh_src, Wh_dst], dim=-1)              # (E, H, 2D)
        attn_raw = (e * self.a.unsqueeze(0)).sum(-1)          # (E, H)
        attn_raw = F.leaky_relu(attn_raw, 0.2)

        # Softmax per destination node
        attn = attn_raw - attn_raw.max()
        attn = torch.exp(attn)
        if edge_weight is not None:
            attn = attn * edge_weight.view(-1, 1)

        # Aggregate
        out  = torch.zeros(N, self.n_heads, self.head_dim, device=x.device)
        norm = torch.zeros(N, self.n_heads, 1, device=x.device) + 1e-8
        idx  = dst.view(-1, 1, 1).expand(-1, self.n_heads, self.head_dim)
        ni   = dst.view(-1, 1, 1).expand(-1, self.n_heads, 1)
        out.scatter_add_(0, idx, Wh_src * attn.unsqueeze(-1))
        norm.scatter_add_(0, ni, attn.unsqueeze(-1))
        out  = out / norm
        out  = self.dropout(F.elu(out.view(N, -1)))

        return self.norm(out + self.W(x)), attn_raw  # residual + attn map


# ─────────────────────────────────────────────────────────────
class PhysicsConstraint(nn.Module):
    """
    Physics-informed penalty: penalise predictions that violate
    CSD theory — if AR1 is high, risk must be elevated.
    Novel contribution — not in any prior GNN climate paper.
    """
    def __init__(self):
        super().__init__()
        self.ar1_weight   = nn.Parameter(torch.tensor(1.0))
        self.var_weight   = nn.Parameter(torch.tensor(0.5))
        self.skew_weight  = nn.Parameter(torch.tensor(0.3))

    def physics_risk(self, features):
        """
        Compute physics-informed risk from CSD statistics.
        features: (N, F) — expects AR1 at index -5, Var at -4, Skew at -3
        """
        ar1  = torch.sigmoid(features[:, -5])  # AR1 → 0..1
        var_ = torch.sigmoid(features[:, -4])
        skew = torch.sigmoid(torch.abs(features[:, -3]))
        return (self.ar1_weight * ar1 +
                self.var_weight * var_ +
                self.skew_weight * skew) / (self.ar1_weight +
                                            self.var_weight +
                                            self.skew_weight)

    def constraint_loss(self, pred_risk, features):
        """
        Physics penalty: if CSD says high risk, prediction must agree.
        """
        phys_risk = self.physics_risk(features)
        # Penalise if model predicts LOW risk when physics says HIGH
        penalty = F.relu(phys_risk - pred_risk).mean()
        return penalty


# ─────────────────────────────────────────────────────────────
class CascadeEWS(nn.Module):
    """
    Full CascadeEWS architecture for Q1 paper.

    Inputs:
      x_seq     : (N, seq_len, F) — node features over time window
      edge_index: (2, E)
      edge_weight: (E,)
      adj_dense : (N, N) normalised adjacency

    Outputs:
      reg_out   : scalar — global anomaly score (regression)
      clf_logits: (1, n_classes) — alert classification
      risk_map  : (N,) — per-node risk score
      cpi       : (N,) — Cascade Propagation Index (novel)
      ews       : (N,) — Early Warning Signal score (novel)
      attn_maps : list of attention weights per GAT layer (for XAI)
    """
    def __init__(self, in_feat=10, hidden=128, seq_len=12,
                 n_nodes=2592, n_classes=3, dropout=0.3, mc_samples=10):
        super().__init__()
        self.seq_len   = seq_len
        self.n_nodes   = n_nodes
        self.mc_samples = mc_samples

        # ── Spatial encoder: 3 GAT layers ─────────────────
        self.gat1 = GraphAttentionLayer(in_feat, hidden, n_heads=4, dropout=dropout)
        self.gat2 = GraphAttentionLayer(hidden,  hidden, n_heads=4, dropout=dropout)
        self.gat3 = GraphAttentionLayer(hidden,  hidden//2, n_heads=2, dropout=dropout)

        # ── Temporal encoder: 2-layer GRU ─────────────────
        self.gru = nn.GRU(hidden//2, hidden, num_layers=2,
                          batch_first=True, dropout=dropout)

        # ── Physics constraint module ──────────────────────
        self.physics = PhysicsConstraint()

        # ── Novel: Cascade Propagation Index params ────────
        self.cpi_theta  = nn.Parameter(torch.tensor(0.7))   # bifurcation threshold
        self.cpi_lambda = nn.Parameter(torch.tensor(2.0))   # sharpness

        # ── Output heads ───────────────────────────────────
        # 1. Regression: global anomaly score
        self.reg_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        # 2. Classification: alert level (n_classes)
        self.clf_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
        # 3. Per-node risk score
        self.risk_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        # 4. Uncertainty head (aleatoric uncertainty estimate)
        self.unc_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Softplus()  # must be positive
        )

        # ── MC Dropout layer ───────────────────────────────
        self.mc_dropout = nn.Dropout(p=0.2)

    def cascade_propagation_index(self, ar1_scores, adj_dense):
        """
        Novel CPI formula (Eq. 1 in paper):
          CPI_i = Σ_j w_ij · σ(λ(ρ_j - θ))
        where ρ_j = AR1, θ = bifurcation threshold, λ = sharpness.
        """
        node_instab = torch.sigmoid(
            self.cpi_lambda * (ar1_scores - self.cpi_theta))
        if adj_dense.is_sparse:
            cpi = torch.sparse.mm(adj_dense, node_instab.unsqueeze(1)).squeeze()
        else:
            cpi = adj_dense @ node_instab
        return cpi  # (N,) higher = more cascade risk

    def early_warning_score(self, x_last):
        """
        EWS score combining AR1, variance, skewness, kurtosis, residual.
        x_last: (N, F) — features at last timestep in window.
        """
        # CSD feature indices: -5=AR1, -4=Var, -3=Skew, -2=Kurt, -1=Resid
        ar1   = torch.sigmoid(self.cpi_lambda * (x_last[:, -5] - self.cpi_theta))
        var_  = torch.sigmoid(x_last[:, -4])
        skew  = torch.sigmoid(torch.abs(x_last[:, -3]))
        kurt  = torch.sigmoid(torch.abs(x_last[:, -2]))
        resid = torch.sigmoid(x_last[:, -1])
        ews = (ar1 + 0.8*var_ + 0.5*skew + 0.3*kurt + 0.4*resid) / 3.0
        return ews.clamp(0, 1)

    def encode_spatial(self, x, edge_index, edge_weight):
        """Run 3 GAT layers, return hidden state + attention maps."""
        attn_maps = []
        h, a1 = self.gat1(x, edge_index, edge_weight)
        attn_maps.append(a1)
        h, a2 = self.gat2(h, edge_index, edge_weight)
        attn_maps.append(a2)
        h, a3 = self.gat3(h, edge_index, edge_weight)
        attn_maps.append(a3)
        return h, attn_maps

    def forward(self, x_seq, edge_index, edge_weight, adj_dense,
                return_attn=False):
        N, T, F = x_seq.shape

        # ── Spatial encoding at each timestep ─────────────
        all_attn = []
        spatial_seq = []
        for t in range(T):
            xt = x_seq[:, t, :]
            h, attn_maps = self.encode_spatial(xt, edge_index, edge_weight)
            spatial_seq.append(h)
            all_attn.append(attn_maps)
        spatial_seq = torch.stack(spatial_seq, dim=1)  # (N, T, hidden//2)

        # ── Temporal encoding ──────────────────────────────
        gru_out, _ = self.gru(spatial_seq)  # (N, T, hidden)
        h_last     = self.mc_dropout(gru_out[:, -1, :])  # (N, hidden)

        # ── Global pooling ─────────────────────────────────
        h_global = h_last.mean(0, keepdim=True)  # (1, hidden)

        # ── Outputs ────────────────────────────────────────
        reg_out    = self.reg_head(h_global).squeeze()
        clf_logits = self.clf_head(h_global)
        risk_map   = self.risk_head(h_last).squeeze()
        uncertainty = self.unc_head(h_global).squeeze()

        # ── Novel metrics ──────────────────────────────────
        ar1_scores = x_seq[:, -1, -5]   # last timestep AR1 values
        cpi = self.cascade_propagation_index(ar1_scores, adj_dense)
        ews = self.early_warning_score(x_seq[:, -1, :])

        # ── Physics constraint ─────────────────────────────
        phys_penalty = self.physics.constraint_loss(risk_map, x_seq[:, -1, :])

        if return_attn:
            return reg_out, clf_logits, risk_map, cpi, ews, uncertainty, phys_penalty, all_attn
        return reg_out, clf_logits, risk_map, cpi, ews, uncertainty, phys_penalty

    def mc_predict(self, x_seq, edge_index, edge_weight, adj_dense, n_samples=None):
        """
        Monte Carlo dropout for uncertainty quantification.
        Runs n_samples forward passes with dropout active.
        Returns: mean prediction + epistemic uncertainty (std).
        """
        if n_samples is None:
            n_samples = self.mc_samples
        self.train()  # enable dropout
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                reg, clf, risk, cpi, ews, unc, _ = self.forward(
                    x_seq, edge_index, edge_weight, adj_dense)
                preds.append({
                    "reg": reg.item(),
                    "clf_prob": F.softmax(clf, -1).squeeze().numpy(),
                    "risk": risk.numpy(),
                    "cpi": cpi.numpy(),
                    "ews": ews.numpy(),
                })
        self.eval()
        reg_vals  = np.array([p["reg"]  for p in preds])
        risk_vals = np.array([p["risk"] for p in preds])
        cpi_vals  = np.array([p["cpi"]  for p in preds])
        clf_vals  = np.array([p["clf_prob"] for p in preds])
        return {
            "reg_mean": reg_vals.mean(),
            "reg_std":  reg_vals.std(),          # epistemic uncertainty
            "risk_mean": risk_vals.mean(0),
            "risk_std":  risk_vals.std(0),
            "cpi_mean":  cpi_vals.mean(0),
            "clf_mean":  clf_vals.mean(0),
        }


# ─────────────────────────────────────────────────────────────
class CascadeSimulator:
    """
    SIR-inspired cascade failure simulator on the climate graph.
    Models how tipping-point instability propagates across connected
    oceanic/atmospheric systems.
    Rocha et al. (2018) — Science Q1 — showed real-world cascade failures.
    """
    def __init__(self, n_nodes, edge_index, edge_weights,
                 spread_rate=0.25, recovery_rate=0.05):
        self.n_nodes      = n_nodes
        self.edge_index   = edge_index
        self.edge_weights = edge_weights
        self.spread_rate  = spread_rate
        self.recovery_rate = recovery_rate

    def simulate(self, initial_risk, n_steps=24, threshold=0.5):
        """
        Simulate cascade propagation from initial risk map.
        initial_risk: (N,) numpy array — starting risk values
        Returns: (n_steps+1, N) — risk evolution over time
        """
        risk    = np.clip(initial_risk.copy(), 0, 1)
        history = [risk.copy()]
        src, dst = self.edge_index[0], self.edge_index[1]

        for step in range(n_steps):
            new_risk = risk.copy()
            # Spread from high-risk nodes
            for i in range(len(src)):
                s, d = src[i], dst[i]
                w = self.edge_weights[i]
                if risk[s] > threshold:
                    spread = self.spread_rate * w * risk[s]
                    new_risk[d] = min(1.0, new_risk[d] + spread)
            # Natural recovery
            new_risk = new_risk - self.recovery_rate * new_risk
            new_risk = np.clip(new_risk, 0, 1)
            risk = new_risk
            history.append(risk.copy())

        cascade_history = np.array(history)  # (n_steps+1, N)
        # Compute cascade metrics
        peak_risk   = cascade_history.max(0)          # (N,)
        cascade_time = np.argmax(cascade_history > threshold, axis=0)  # (N,)
        total_affected = (cascade_history > threshold).any(0).sum()
        return cascade_history, peak_risk, cascade_time, int(total_affected)


# ─────────────────────────────────────────────────────────────
class CombinedLoss(nn.Module):
    """
    Multi-objective loss for Q1-level training.
    Combines: regression + classification + physics + node risk
    """
    def __init__(self, n_classes=3, lambda_phys=0.1, lambda_risk=0.3):
        super().__init__()
        self.mse         = nn.MSELoss()
        self.ce          = nn.CrossEntropyLoss(
            weight=torch.FloatTensor([1.0, 2.0, 4.0]))  # upweight rare events
        self.lambda_phys = lambda_phys
        self.lambda_risk = lambda_risk

    def forward(self, reg_pred, reg_true, clf_pred, clf_true,
                risk_pred, risk_true, phys_penalty):
        l_reg  = self.mse(reg_pred.unsqueeze(0), reg_true)
        l_clf  = self.ce(clf_pred, clf_true)
        # Normalise risk target to [0,1]
        rt = (risk_true - risk_true.min()) / (risk_true.max() - risk_true.min() + 1e-8)
        l_risk = self.mse(risk_pred, rt)
        l_phys = phys_penalty
        total  = l_reg + l_clf + self.lambda_risk * l_risk + self.lambda_phys * l_phys
        return total, {"reg": l_reg.item(), "clf": l_clf.item(),
                       "risk": l_risk.item(), "phys": l_phys.item()}
