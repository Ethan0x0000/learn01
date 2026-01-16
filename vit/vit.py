import torch
from torch import nn
from modules import PatchEmbed, EncoderBlock


class ViTEmbeddings(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

    def forward(self, x):
        # x: (B, N, D)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=1000,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.embeddings = ViTEmbeddings(num_patches, embed_dim)

        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.embeddings(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)
