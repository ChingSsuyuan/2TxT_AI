import torch
import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


class CLIPCaptioner(nn.Module):
    def __init__(self, clip_model, vocab_size, embed_dim=512, hidden_dim=512):
        super().__init__()
        self.clip_model = clip_model
        for param in self.clip_model.parameters():
            param.requires_grad = False
            

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8), 
            num_layers=6
        )
        self.fc = nn.Linear(embed_dim, vocab_size)
        
