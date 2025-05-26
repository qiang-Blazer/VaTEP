import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False).to('cuda')
        return self.dropout(x)


# 残差3D卷积块
class ResConv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResConv3dBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.adjust_channels = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.adjust_channels(x)
        x = F.leaky_relu(self.bn(self.conv(x)), 0.2)
        return x + residual


class VideoEncoder(nn.Module):
    def __init__(self, C=256):
        super(VideoEncoder, self).__init__()
        # 多尺度卷积
        self.resnet1 = ResConv3dBlock(1, C//8, kernel_size=(5, 5, 5), padding=(2, 2, 2))
        self.resnet2 = ResConv3dBlock(C//8, C//4, kernel_size=(5, 5, 5), padding=(2, 2, 2))
        self.resnet3 = ResConv3dBlock(C//4, C//2, kernel_size=(5, 5, 5), padding=(2, 2, 2))
        self.resnet4 = ResConv3dBlock(C//2, C, kernel_size=(5, 5, 5), padding=(2, 2, 2))
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) # 降空间维度
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=C, nhead=4, dim_feedforward=C*2, batch_first=True), num_layers=2
        )
        self.pe = PositionalEncoding(C, 0.1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, C))

    def forward(self, x):
        # x: (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.pool(self.resnet1(x))
        x = self.pool(self.resnet2(x))
        x = self.pool(self.resnet3(x))
        x = self.pool(self.resnet4(x))  # (B, D, T, H', W')
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)  # (B, D, T)
        x = x.permute(0, 2, 1)  # (B, T, D)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, D)
        x = self.pe(x)  # (B, T+1, D)
        x = self.transformer(x)  # (B, T+1, D)
        return x


class VideoDecoder(nn.Module):
    def __init__(self, D=256, C=1):
        super().__init__()
        # 转置卷积层
        self.C = C
        self.up = nn.Linear(D, D*7*7)
        self.deconv1 = nn.ConvTranspose2d(D, D//2, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(D//2, D//4, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(D//4, D//8, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(D//8, C, 4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入: (B, T, D)
        B, T, D = x.shape
        x = x.view(B * T, D)  # 重塑为 (B*T, D)
        x = self.up(x)  # (B*T, D*7*7)
        x = x.view(B*T, D, 7, 7)  # (B*T, D, 7, 7)
        x = self.relu(self.deconv1(x))  # (B*T, D//2, 14, 14)
        x = self.relu(self.deconv2(x))  # (B*T, D//4, 28, 28)
        x = self.relu(self.deconv3(x))  # (B*T, D//8, 56, 56)
        x = self.deconv4(x)  # (B*T, C, 112, 112)
        x = self.sigmoid(x)  # (B*T, C, 112, 112)
        x = x.view(B, T, self.C, 112, 112)  # (B, T, C, 112, 112)
        return x


class FrameClassifier(nn.Module):
    def __init__(self, D=256, K=11):
        super().__init__()
        self.tcn = nn.Conv1d(D, D, kernel_size=3, padding=1)
        self.mlp = nn.Sequential(
            nn.Linear(D, D//2),
            nn.ReLU(),
            nn.Linear(D//2, D//4),
            nn.ReLU(),            
            nn.Linear(D//4, K)
        )

    def forward(self, x):
        # x: (B, T, D)
        x = x.transpose(1, 2)  # (B, D, T)
        x = F.relu(self.tcn(x))  # (B, D, T)
        x = x.transpose(1, 2)  # (B, T, D)
        x = self.mlp(x)  # (B, T, K)
        return x



# 示例用法
if __name__ == "__main__":
    encoder = VideoEncoder().cuda()
    input_video = torch.randn(2, 48, 1, 112, 112).cuda()
    embedding = encoder(input_video)
    print(embedding.shape)   

    decoder = VideoDecoder().cuda()
    output_video = decoder(embedding)
    print(output_video.shape)

    classifier = FrameClassifier().cuda()
    frame_scores = classifier(embedding)
    print(frame_scores.shape)
