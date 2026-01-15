import torch
from modules import PatchEmbed
from vit import ViTEmbeddings

def test_patch_embed():
    x = torch.randn(2, 3, 224, 224)
    pe = PatchEmbed()
    out = pe(x)
    assert out.shape == (2, 196, 768)

def test_pos_embed():
    x = torch.randn(2, 196, 768)
    emb = ViTEmbeddings(num_patches=196, embed_dim=768)
    out = emb(x)
    assert out.shape == (2, 197, 768)

test_patch_embed()
test_pos_embed()
print("✅ Test[1] passed")
