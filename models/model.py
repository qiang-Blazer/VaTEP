import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_pretrain import VideoEncoder



class VaTEP(nn.Module):
    def __init__(self, C=256, N=12):
        super().__init__()
        self.encoder = VideoEncoder(C)

        self.categorical_embedding1 = nn.Embedding(2, C)
        self.categorical_embedding2 = nn.Embedding(2, C)
        self.categorical_embedding3 = nn.Embedding(2, C)
        self.continuous_embedding = nn.Linear(1, C)
        self.table_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=C, nhead=4, dim_feedforward=C*2, batch_first=True), num_layers=4
        )
        self.cross_attention = nn.MultiheadAttention(embed_dim=C, num_heads=1, batch_first=True)
        self.linearFH = nn.Sequential(nn.Linear(C*N, C), nn.ReLU(), nn.Dropout(0.5), nn.Linear(C, C//2), nn.ReLU(), nn.Linear(C//2, 2))
        self.linearMP = nn.Sequential(nn.Linear(C*N, C), nn.ReLU(), nn.Dropout(0.5), nn.Linear(C, C//2), nn.ReLU(), nn.Linear(C//2, 3))
        self.linearLB = nn.Sequential(nn.Linear(C*N, C), nn.ReLU(), nn.Dropout(0.5), nn.Linear(C, C//2), nn.ReLU(), nn.Linear(C//2, 3))

    def forward(self, x, table_cat, table_con): # table_cat: [B,3], table_con: [B,K-3]
        B,N,T,H,W = x.shape
        x = x.reshape(B*N,T,1,H,W)
        x = self.encoder(x) #[B*N,T+1,D]
        x = x[:,1,:].reshape(B,N,-1) #[B,N,D]

        table_cat1 = table_cat[:,0]-1 #[B,1]
        table_cat1 = self.categorical_embedding1(table_cat1) #[B,1,D]
        table_cat2 = table_cat[:,1] #[B,1]
        table_cat2 = self.categorical_embedding2(table_cat2) #[B,1,D]
        table_cat3 = table_cat[:,2] #[B,1]
        table_cat3 = self.categorical_embedding3(table_cat3) #[B,1,D]
        table_con = self.continuous_embedding(table_con.unsqueeze(-1)) #[B,K-3,D]
        table = torch.cat((table_cat1.unsqueeze(1), table_cat2.unsqueeze(1), table_cat3.unsqueeze(1), table_con), dim=1) #[B,K,D]
        table = self.table_transformer(table) #[B,K,D]

        attn_output, attn_weights = self.cross_attention(x, table, table)
        x = x + attn_output #[B,N,D]
        x = x.reshape(B,-1) #[B,N*D]
        FH = self.linearFH(x) #[B,2]
        MP = self.linearMP(x) #[B,3]
        LB = self.linearLB(x) #[B,3]

        return FH, MP, LB


if __name__ == '__main__':
    video_tensor = torch.rand(2, 12, 32, 112, 112).to('cuda')
    table_categorical = torch.randint(0, 2, (2, 3)).to('cuda')
    table_continuous = torch.randn(2, 38).to('cuda')
    model = VaTEP(C=256, N=12).to('cuda')
    HF,MP,LB = model(video_tensor, table_categorical, table_continuous)
    print(HF.shape, MP.shape, LB.shape)

    # label = torch.randint(0, 2, (2,)).to('cuda')
    # loss_fn = nn.CrossEntropyLoss()
    # loss = loss_fn(output, label)
    # print(loss)