import os
import numpy as np
import torch
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader


from dataloader.dataset_recon import FrameAndPhase
from models.model_pretrain import VideoEncoder, VideoDecoder, FrameClassifier
from utils.utils import set_seed, make_runs_dir, save_args



def compute_batch_class_weights(labels, num_classes):
    """计算当前批次的类别权重"""
    class_counts = torch.bincount(labels.view(-1), minlength=num_classes)
    class_weights = 1.0 / (class_counts.float() + 1e-6)  # 避免除零
    class_weights[torch.where(class_counts == 0)] = 0 # batch中没有该类别的样本，则权重为0
    class_weights = class_weights / torch.sum(class_weights)  # 归一化

    return class_weights


def monotonic_constraint_loss(probs: torch.Tensor) -> torch.Tensor:
    """
    保证每个时间步最大概率类别序号非递减的正则化损失项，支持批量输入。
    
    Args:
        probs (torch.Tensor): 形状为[B, T, C]的概率张量，B为批量大小，T为时间步，C为类别数。
        
    Returns:
        torch.Tensor: 正则化损失值（标量）。
    """
    B, T, C = probs.shape
    # 生成类别索引：0到C-1
    categories = torch.arange(C, dtype=torch.float, device=probs.device)  # 形状 [C]
    # 计算每个时间步的期望类别（soft argmax）
    probs = F.gumbel_softmax(probs, tau=0.5, hard=False)  # 形状 [B, T, C]，每个元素表示对应类别的概率, tau越小越接近one-hot编码, hard=True给出one-hot编码
    expected_categories = torch.sum(probs * categories.view(1, 1, C), dim=2)  # 形状 [B, T]
    # 计算相邻时间步的期望差值
    deltas = expected_categories[:, 1:] - expected_categories[:, :-1]  # 形状 [B, T-1]
    # 对期望值下降的部分进行惩罚（差值小于0）
    loss = torch.sum(torch.clamp(-deltas, min=0.0), dim=1)  # 形状 [B]
    # 对所有样本的损失取平均
    loss = torch.mean(loss)
    return loss


def train(epoch):
    running_recon_loss = 0.0
    running_acc = 0.0
    # train the encoder and decoder
    encoder.train() 
    decoder.train()
    for video_tensors,labels in train_dataloader:  # [B, T, H, W], [B, T]
        optimizer_recon.zero_grad()
        video_tensors = video_tensors.unsqueeze(2).cuda() # [B, T, 1, H, W]
        embeddings = encoder(video_tensors)[:, 1:, :] # [B, T, D]
        recons = decoder(embeddings) # [B, T, 1, H, W]
        recon_loss = criterion_mse(recons, video_tensors) 
        running_recon_loss += recon_loss.item() 

        recon_loss.backward()        
        optimizer_recon.step()
    scheduler_recon.step()  
    # train the classifier
    encoder.train() 
    decoder.eval()
    classifier.train()
    for video_tensors,labels in train_dataloader:  # [B, T, H, W], [B, T]
        video_tensors = video_tensors.unsqueeze(2).cuda() # [B, T, 1, H, W]
        
        embeddings = encoder(video_tensors)[:, 1:, :] # [B, T, D]
        recons = decoder(embeddings).detach() # [B, T, 1, H, W]
        embeddings2 = encoder(recons)[:, 1:, :] # [B, T, D]

        preds = classifier(embeddings) # [B, T, 11]
        preds2 = classifier(embeddings2)
        # loss_monotonic = monotonic_constraint_loss(probs=preds)
        # loss_monotonic2 = monotonic_constraint_loss(probs=preds2)
        preds = preds.permute(0,2,1) # [B, 11, T]
        preds2 = preds2.permute(0,2,1) # [B, 11, T]
        B, C, T = preds.shape
        class_weights = compute_batch_class_weights(labels, num_classes=11)
        criterion_ce.weight = class_weights.cuda()
        cls_loss = criterion_ce(preds, labels.cuda()) + criterion_ce(preds2, labels.cuda()) #+ 0.01*loss_monotonic + 0.01*loss_monotonic2
        acc = ((torch.argmax(preds,  dim=1).cpu() == labels).sum().item() + 
               (torch.argmax(preds2, dim=1).cpu() == labels).sum().item()) / (2*B*T)
        running_acc += acc
        cls_loss.backward()
        optimizer_cls.step()
    scheduler_cls.step()

    recon_loss = running_recon_loss/len(train_dataloader)
    acc = running_acc/len(train_dataloader)
    log = f'Epoch: {epoch:03d}, [Train] Recon loss: {recon_loss:.4f} Acc: {acc:.4f}' 
    print(log)
    global training_logs
    training_logs += log+"\n"


def val(epoch):
    running_recon_loss = 0.0
    running_acc = 0.0
    encoder.eval()
    decoder.eval()
    classifier.eval()
    with torch.no_grad():
        for video_tensors,labels in val_dataloader:  # [B, T, H, W], [B, T]
            video_tensors = video_tensors.unsqueeze(2).cuda()
            embeddings = encoder(video_tensors)[:, 1:, :] # [B, T, D]
            recons = decoder(embeddings) # [B, T, 1, H, W]
            preds = classifier(embeddings) # [B, T, 11]
            preds = preds.permute(0,2,1) # [B, 12, T]
            B, C, T = preds.shape
            recon_loss = criterion_mse(recons, video_tensors) 
            running_recon_loss += recon_loss.item() 
            acc = (torch.argmax(preds, dim=1).cpu() == labels).sum().item() / (B*T)
            running_acc += acc

    recon_loss = running_recon_loss/len(val_dataloader)
    acc = running_acc/len(val_dataloader)
    log = f'Epoch: {epoch:03d}, [Val] Recon loss: {recon_loss:.4f} Acc: {acc:.4f}' 
    print(log)
    global training_logs
    training_logs += log+"\n"  

    return recon_loss, acc

if __name__ == '__main__':
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=str, default='0', help='Cuda id to use')
    parser.add_argument('--epoch', type=int, default=1000, help='Number of epochs') 
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--scheduler', type=str, default="cosineannealinglr", help='Scheduler used in training')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--img_size', type=int, default=112, help='Frame height or width')
    parser.add_argument('--frame_num', type=int, default=48, help='Number of frames per video')

    args = parser.parse_args()
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_id)

    # Data
    train_dataset = FrameAndPhase(frame_num=args.frame_num, img_size=args.img_size, seed=args.seed, train=True)
    val_dataset = FrameAndPhase(frame_num=args.frame_num, img_size=args.img_size, seed=args.seed, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    encoder = VideoEncoder(C=256)
    decoder = VideoDecoder(D=256, C=1)
    classifier = FrameClassifier(D=256, K=11)
    encoder.cuda()
    decoder.cuda()
    classifier.cuda()

    # Loss Fuctions
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    criterion_ce.cuda()
    criterion_mse.cuda()

    # Optimizer
    optimizer_recon = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=args.lr)
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    #  Scheduler
    if args.scheduler.lower() == "cosineannealinglr":
        scheduler_recon = CosineAnnealingLR(optimizer_recon, T_max=20) #drop to 0 at T_max/3*T_max/5*T_max/7*T_max , restart at 2*T_max/4*T_max/6*T_max/8*T_max 
        scheduler_cls = CosineAnnealingLR(optimizer_cls, T_max=20) #drop to 0 at T_max/3*T_max/5*T_max/7*T_max , restart at 2*T_max/4*T_max/6*T_max/8*T_max 
    elif args.scheduler.lower() == "cosineannealingwarmrestarts":
        scheduler_recon = CosineAnnealingWarmRestarts(optimizer_recon, T_0=20, T_mult=2, eta_min=0) #drop to 0 and restart at 20/60/140/300 
        scheduler_cls = CosineAnnealingWarmRestarts(optimizer_cls, T_0=20, T_mult=2, eta_min=0) #drop to 0 and restart at 20/60/140/300 
    else:
        raise ValueError(f"The argument --scheduler can not be {args.scheduler}")

    #train the model
    result_dir = make_runs_dir('runs','pretrain')
    training_logs = ""
    print("Start training...")
    max_acc = 0.0
    min_recon_loss = 10000
    count = 0
    for epoch in range(1,args.epoch+1):
        train(epoch)           
        recon_loss, acc = val(epoch)
        torch.save(encoder.state_dict(), os.path.join(result_dir, 'encoder.pth'))
        if acc > max_acc and recon_loss < min_recon_loss:
            count = 0
            max_acc = acc
            min_recon_loss = recon_loss
        else:
            count += 1
            if count == 10:
                break
    #make dir to save results 
    save_args(args, result_dir)
    with open(os.path.join(result_dir,'logs.txt'), 'w') as f:
        f.write(training_logs)
        f.close()