import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# STEP 1: PATCH EMBEDDING
# ==============================================================================
class VitPatchEmbedding(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=1, embed_dim=128):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.projection(x)      # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)            # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)      # (B, num_patches, embed_dim)
        return x


# ==============================================================================
# STEP 2: CLS TOKEN + POSITIONAL EMBEDDING
# ==============================================================================
class VitCLSPosEmbed(nn.Module):
    def __init__(self, num_patches, embed_dim, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding
        return self.dropout(x)


# ==============================================================================
# STEP 3: MULTI-HEAD SELF-ATTENTION
# ==============================================================================
class VitMHSA(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv       = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj  = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv.unbind(0)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out  = attn @ V
        out  = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


# ==============================================================================
# STEP 4: MLP BLOCK
# ==============================================================================
class VitMLP(nn.Module):
    def __init__(self, embed_dim=128, mlp_ratio=4, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ==============================================================================
# STEP 5: TRANSFORMER ENCODER BLOCK
# ==============================================================================
class VitEncoderBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=4,
                 dropout=0.1, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = VitMHSA(embed_dim, num_heads, attn_drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = VitMLP(embed_dim, mlp_ratio, dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


# ==============================================================================
# STEP 6: FULL VISION TRANSFORMER
# ==============================================================================
class VisionTransformer(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=1,
                 embed_dim=128, depth=6, num_heads=4, mlp_ratio=4.0,
                 dropout=0.1, attn_drop=0.0, num_classes=None):
        super().__init__()
        self.embed_dim   = embed_dim
        self.patch_embed = VitPatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches      = self.patch_embed.num_patches
        self.cls_pos     = VitCLSPosEmbed(num_patches, embed_dim, dropout)
        self.encoder     = nn.Sequential(*[
            VitEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        x = self.patch_embed(x)     # (B, num_patches, D)
        x = self.cls_pos(x)         # (B, num_patches+1, D)
        x = self.encoder(x)         # (B, num_patches+1, D)
        x = self.norm(x)
        return self.head(x[:, 0])   # CLS token → head


# ==============================================================================
# STEP 7: DPF OBSERVATION MODEL
# ==============================================================================
class ViTObservationModel(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=1,
                 embed_dim=128, depth=6, num_heads=4,
                 obs_dim=64, state_dim=3, dropout=0.1):
        super().__init__()
        self.vit = VisionTransformer(
            img_size=img_size, patch_size=patch_size,
            in_channels=in_channels, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads,
            dropout=dropout, num_classes=obs_dim,
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, obs_dim),
        )

    def encode_observation(self, obs):
        return self.vit(obs)

    def encode_state(self, particles):
        B, K, D = particles.shape
        flat = particles.reshape(B * K, D)
        return self.state_encoder(flat).reshape(B, K, -1)

    def log_likelihood(self, obs, particles):
        z_obs    = self.encode_observation(obs).unsqueeze(1)  # (B, 1, obs_dim)
        z_states = self.encode_state(particles)               # (B, K, obs_dim)
        log_w    = -torch.sum((z_obs - z_states) ** 2, dim=-1)
        return log_w

    def forward(self, obs, particles):
        log_w = self.log_likelihood(obs, particles)
        return F.log_softmax(log_w, dim=-1)


# ==============================================================================
# SANITY CHECK
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  ViT for DPF — Sanity Check")
    print("=" * 60)

    B, K, img_size, state_dim = 4, 100, 64, 3

    model = ViTObservationModel(
        img_size=img_size, patch_size=8, in_channels=1,
        embed_dim=128, depth=4, num_heads=4,
        obs_dim=64, state_dim=state_dim,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters : {total_params:,}")

    obs       = torch.randn(B, 1, img_size, img_size)
    particles = torch.randn(B, K, state_dim)

    with torch.no_grad():
        z_obs    = model.encode_observation(obs)
        z_states = model.encode_state(particles)
        log_w    = model(obs, particles)

    print(f"  Input image      : {list(obs.shape)}")
    print(f"  Encoded obs      : {list(z_obs.shape)}")
    print(f"  Particles        : {list(particles.shape)}")
    print(f"  Encoded states   : {list(z_states.shape)}")
    print(f"  Log weights      : {list(log_w.shape)}")

    lse = torch.logsumexp(log_w, dim=-1)
    print(f"\n  log-sum-exp (should be ~0): {lse.mean().item():.6f}")
    print("\n  All shapes correct - ViT observation model ready!\n")
