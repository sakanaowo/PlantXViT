import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights


# inception block (chỉnh sửa: tổng output channels = 512)
class InceptionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 192, kernel_size=1)  # Tăng lên 192

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 160, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(160, 160, kernel_size=(3, 1), padding=(1, 0)),
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 160, kernel_size=1),  # Tăng lên 160
        )

    def forward(self, x):
        b1 = self.branch1x1(x)
        b2 = self.branch3x3(x)
        b3 = self.branch_pool(x)
        return torch.cat([b1, b2, b3], dim=1)  # Output shape: (B, 512, H, W)


# patch embedding: split patch -> Linear
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size=5, emb_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.proj = nn.Linear(in_channels * patch_size * patch_size, emb_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, -1, C * self.patch_size * self.patch_size)
        return self.proj(x)  # shape: (b,num patches,emb size)


# -------- Transformer Encoder Block (ViT block) --------
class TransformerBlock(nn.Module):
    def __init__(self, emb_size=16, dropout=0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, num_heads=2, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size * 2),
            nn.GELU(),
            nn.Linear(emb_size * 2, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_attn, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + x_attn
        x = x + self.mlp(self.norm2(x))
        return x


# -------- PlantXViT Model --------
class PlantXViT(nn.Module):
    def __init__(self, num_classes=4, patch_size=5, emb_size=16, num_blocks=4, dropout=0.1):
        super().__init__()

        # VGG16 (2 blocks)
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
        self.vgg_block = nn.Sequential(*list(vgg.features[:10]))
        # self.vgg_block = nn.Sequential(*vgg[:10])  # output: (B, 128, 56, 56)

        # Inception-like block → (B, 384, 56, 56)
        self.inception = InceptionBlock(in_channels=128)

        # Patch Embedding → (B, 121, 16)
        # self.patch_embed = PatchEmbedding(in_channels=384, patch_size=patch_size, emb_size=emb_size)
        self.patch_embed = PatchEmbedding(in_channels=512, patch_size=patch_size, emb_size=emb_size)

        # Transformer blocks
        self.transformer = nn.Sequential(*[TransformerBlock(emb_size, dropout) for _ in range(num_blocks)])

        # Classification head
        self.norm = nn.LayerNorm(emb_size)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # (B, emb_size, 1)
        self.classifier = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.vgg_block(x)  # (B, 128, 56, 56)
        x = self.inception(x)  # (B, 384, 56, 56)
        x = self.patch_embed(x)  # (B, 121, 16)
        x = self.transformer(x)  # (B, 121, 16)
        x = self.norm(x)  # (B, 121, 16)
        x = x.permute(0, 2, 1)  # (B, 16, 121)
        x = self.global_pool(x).squeeze(-1)  # (B, 16)
        return self.classifier(x)  # (B, num_classes)
