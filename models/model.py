import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import sys
sys.path.append('f:\Task\codes\VaTEP')
from models.model_pretrain import VideoEncoder



class VaTEP(nn.Module):
    def __init__(self, C=1, D=256, N=12):
        super().__init__()
        self.encoder = VideoEncoder(C, D)

        self.categorical_embedding1 = nn.Embedding(2, D)
        self.categorical_embedding2 = nn.Embedding(2, D)
        self.categorical_embedding3 = nn.Embedding(2, D)
        self.continuous_embedding = nn.Linear(1, D)
        self.table_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=D, nhead=4, dim_feedforward=D*2, batch_first=True), num_layers=4
        )
        self.cross_attention = nn.MultiheadAttention(embed_dim=D, num_heads=4, batch_first=True)
        self.layernorm_video = nn.LayerNorm(D)
        self.layernorm_table = nn.LayerNorm(D)
        self.linearFH = nn.Sequential(nn.Linear(D*N, D), nn.ReLU(), nn.Dropout(0.5), nn.Linear(D, D//2), nn.ReLU(), nn.Linear(D//2, 2))
        self.linearMP = nn.Sequential(nn.Linear(D*N, D), nn.ReLU(), nn.Dropout(0.5), nn.Linear(D, D//2), nn.ReLU(), nn.Linear(D//2, 3))
        self.linearLB = nn.Sequential(nn.Linear(D*N, D), nn.ReLU(), nn.Dropout(0.5), nn.Linear(D, D//2), nn.ReLU(), nn.Linear(D//2, 3))

    def forward(self, x, table_cat, table_con): # table_cat: [B,3], table_con: [B,K-3]
        B,N,T,H,W = x.shape
        x = x.reshape(B*N,T,1,H,W)
        x = self.encoder(x) #[B*N,T+1,D]
        x = x[:,1,:].reshape(B,N,-1) #[B,N,D]
        x = self.layernorm_video(x)

        table_cat1 = table_cat[:,0]-1 #[B,1]
        table_cat1 = self.categorical_embedding1(table_cat1) #[B,1,D]
        table_cat2 = table_cat[:,1] #[B,1]
        table_cat2 = self.categorical_embedding2(table_cat2) #[B,1,D]
        table_cat3 = table_cat[:,2] #[B,1]
        table_cat3 = self.categorical_embedding3(table_cat3) #[B,1,D]
        table_con = self.continuous_embedding(table_con.unsqueeze(-1)) #[B,K-3,D]
        table = torch.cat((table_cat1.unsqueeze(1), table_cat2.unsqueeze(1), table_cat3.unsqueeze(1), table_con), dim=1) #[B,K,D]
        table = self.table_transformer(table) #[B,K,D]
        table = self.layernorm_table(table)

        attn_output, attn_weights = self.cross_attention(x, table, table)
        x = x + attn_output #[B,N,D]
        x = x.reshape(B,-1) #[B,N*D]
        FH = self.linearFH(x) #[B,2]
        MP = self.linearMP(x) #[B,3]
        LB = self.linearLB(x) #[B,3]

        return FH, MP, LB


if __name__ == '__main__':
    video_tensor = torch.rand(2, 12, 32, 112, 112)
    table_categorical = torch.cat((torch.randint(1, 3, (2,1)), torch.randint(0, 2, (2, 2))), dim=1)
    table_continuous = torch.randn(2, 36)
    model = VaTEP(C=1, D=256, N=12)
    # HF,MP,LB = model(video_tensor, table_categorical, table_continuous)
    # print(HF.shape, MP.shape, LB.shape)

    # label = torch.randint(0, 2, (2,)).to('cuda')
    # loss_fn = nn.CrossEntropyLoss()
    # loss = loss_fn(output, label)
    # print(loss)

    result = summary(model, 
        input_data=[video_tensor, table_categorical, table_continuous],
        dtypes=[torch.float32, torch.int64, torch.float32], 
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
        device="cpu",
        verbose=1)
    
    with open("model_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"模型结构：\n{result}\n")
        f.write(f"总参数：{result.total_params:,}\n")
        f.write(f"可训练参数：{result.trainable_params:,}")

    print("已保存到 model_summary.txt")