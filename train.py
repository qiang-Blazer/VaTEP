import os
import numpy as np
import random
import torch
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
import albumentations as A
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ExponentialLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from dataloader.dataset import FrameAndTable
from models.model import VaTEP
from utils.utils import set_seed, make_runs_dir, save_args, get_auc_and_threshold



def train(epoch):
    running_loss_FH = 0.0
    running_loss_MP = 0.0
    running_loss_LB = 0.0
    all_labels_FH = []
    all_preds_FH = []
    all_labels_MP = []
    all_preds_MP = []
    all_labels_LB = []
    all_preds_LB = []
    model.train() 
    for video_tensors, table_cat, table_con, FetalHeart_label, Multiple_label, LiveBirth_label in train_dataloader:
        optimizer.zero_grad()
        FH,MP,LB = model(video_tensors.cuda(), table_cat.cuda(), table_con.cuda())
        masks = (FetalHeart_label == 1).squeeze(-1)
        MP = MP[masks][:,1:]
        LB = LB[masks][:,1:]
        Multiple_label = Multiple_label[masks]
        LiveBirth_label = LiveBirth_label[masks]
        loss_FH = criterion(FH,FetalHeart_label.squeeze(-1).cuda())
        loss_MP = criterion(MP,Multiple_label.squeeze(-1).cuda())
        loss_LB = criterion(LB,LiveBirth_label.squeeze(-1).cuda())
        loss = loss_FH + loss_MP + loss_LB
        running_loss_FH += loss_FH.item()
        running_loss_MP += loss_MP.item()
        running_loss_LB += loss_LB.item()
        all_labels_FH.extend(FetalHeart_label.squeeze(-1).detach().numpy().tolist())
        all_preds_FH.extend(torch.softmax(FH, 1)[:,1].detach().cpu().numpy().tolist())
        all_labels_MP.extend(Multiple_label.squeeze(-1).detach().numpy().tolist())
        all_preds_MP.extend(torch.softmax(MP, 1)[:,1].detach().cpu().numpy().tolist())
        all_labels_LB.extend(LiveBirth_label.squeeze(-1).detach().numpy().tolist())
        all_preds_LB.extend(torch.softmax(LB, 1)[:,1].detach().cpu().numpy().tolist())

        loss.backward()        
        optimizer.step()
    scheduler.step()

    all_labels_FH = np.array(all_labels_FH) 
    all_preds_FH = np.array(all_preds_FH)
    auc_FH,threshold,fpr,tpr = get_auc_and_threshold(all_labels_FH, all_preds_FH)  
    acc_FH = accuracy_score(all_labels_FH, all_preds_FH>threshold)  
    all_labels_MP = np.array(all_labels_MP) 
    all_preds_MP = np.array(all_preds_MP)
    auc_MP,threshold,fpr,tpr = get_auc_and_threshold(all_labels_MP, all_preds_MP)  
    acc_MP = accuracy_score(all_labels_MP, all_preds_MP>threshold)  
    all_labels_LB = np.array(all_labels_LB) 
    all_preds_LB = np.array(all_preds_LB)
    auc_LB,threshold,fpr,tpr = get_auc_and_threshold(all_labels_LB, all_preds_LB)  
    acc_LB = accuracy_score(all_labels_LB, all_preds_LB>threshold)  

    loss_FH = running_loss_FH/len(train_dataloader)
    loss_MP = running_loss_MP/len(train_dataloader)
    loss_LB = running_loss_LB/len(train_dataloader) 
    log = f'Epoch: {epoch:03d}, [Train] Loss - FH: {loss_FH:.4f} MP: {loss_MP:.4f} LB: {loss_LB:.4f} Acc - FH: {acc_FH:.4f} MP: {acc_MP:.4f} LB: {acc_LB:.4f} Auc - FH: {auc_FH:.4f} MP: {auc_MP:.4f} LB: {auc_LB:.4f}' 
    print(log)
    global training_logs
    training_logs += log+"\n"


def val(epoch):
    running_loss_FH = 0.0
    running_loss_MP = 0.0
    running_loss_LB = 0.0
    all_labels_FH = []
    all_preds_FH = []
    all_labels_MP = []
    all_preds_MP = []
    all_labels_LB = []
    all_preds_LB = []
    model.eval() 
    with torch.no_grad():
        for video_tensors, table_cat, table_con, FetalHeart_label, Multiple_label, LiveBirth_label in val_dataloader:
            FH,MP,LB = model(video_tensors.cuda(), table_cat.cuda(), table_con.cuda())
            masks = (FetalHeart_label == 1).squeeze(-1)
            MP = MP[masks]
            LB = LB[masks]
            Multiple_label = Multiple_label[masks]
            LiveBirth_label = LiveBirth_label[masks]
            loss_FH = criterion(FH,FetalHeart_label.squeeze(-1).cuda())
            loss_MP = criterion(MP,Multiple_label.squeeze(-1).cuda())
            loss_LB = criterion(LB,LiveBirth_label.squeeze(-1).cuda())
            running_loss_FH += loss_FH.item()
            running_loss_MP += loss_MP.item()
            running_loss_LB += loss_LB.item()
            all_labels_FH.extend(FetalHeart_label.squeeze(-1).detach().numpy().tolist())
            all_preds_FH.extend(torch.softmax(FH, 1)[:,1].detach().cpu().numpy().tolist())
            all_labels_MP.extend(Multiple_label.squeeze(-1).detach().numpy().tolist())
            all_preds_MP.extend(torch.softmax(MP, 1)[:,1].detach().cpu().numpy().tolist())
            all_labels_LB.extend(LiveBirth_label.squeeze(-1).detach().numpy().tolist())
            all_preds_LB.extend(torch.softmax(LB, 1)[:,1].detach().cpu().numpy().tolist())

    all_labels_FH = np.array(all_labels_FH) 
    all_preds_FH = np.array(all_preds_FH)
    auc_FH,threshold,fpr,tpr = get_auc_and_threshold(all_labels_FH, all_preds_FH)  
    acc_FH = accuracy_score(all_labels_FH, all_preds_FH>threshold)  
    all_labels_MP = np.array(all_labels_MP) 
    all_preds_MP = np.array(all_preds_MP)
    auc_MP,threshold,fpr,tpr = get_auc_and_threshold(all_labels_MP, all_preds_MP)  
    acc_MP = accuracy_score(all_labels_MP, all_preds_MP>threshold)  
    all_labels_LB = np.array(all_labels_LB) 
    all_preds_LB = np.array(all_preds_LB)
    auc_LB,threshold,fpr,tpr = get_auc_and_threshold(all_labels_LB, all_preds_LB)  
    acc_LB = accuracy_score(all_labels_LB, all_preds_LB>threshold)  

    loss_FH = running_loss_FH/len(val_dataloader)
    loss_MP = running_loss_MP/len(val_dataloader)
    loss_LB = running_loss_LB/len(val_dataloader) 
    log = f'Epoch: {epoch:03d}, [Val] Loss - FH: {loss_FH:.4f} MP: {loss_MP:.4f} LB: {loss_LB:.4f} Acc - FH: {acc_FH:.4f} MP: {acc_MP:.4f} LB: {acc_LB:.4f} Auc - FH: {auc_FH:.4f} MP: {auc_MP:.4f} LB: {auc_LB:.4f}' 
    print(log)
    global training_logs
    training_logs += log+"\n"

    return auc_FH


if __name__ == '__main__':
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=str, default='0', help='Cuda id to use')
    parser.add_argument('--epoch', type=int, default=50, help='Number of epochs') 
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--scheduler', type=str, default="ExponentialLR", help='Scheduler used in training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--img_size', type=int, default=112, help='Frame height or width')
    parser.add_argument('--frame_num', type=int, default=48, help='Number of extracted frames per video')
    parser.add_argument('--n', type=int, default=6, help='Extraction times')
    parser.add_argument('--pretrained_weights', type=str, default='runs/pretrain/encoder.pth', help='Pretrained model weights')

    args = parser.parse_args()
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_id)
    # Data
    train_dataset = FrameAndTable(frame_num=args.frame_num,  n=args.n, img_size=args.img_size, seed=args.seed, train=True)
    val_dataset = FrameAndTable(frame_num=args.frame_num, n=args.n, img_size=args.img_size, seed=args.seed, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)
    # Model
    model = VaTEP(C=256, N=args.n*2)
    if args.pretrained_weights is not None:
        model.encoder.load_state_dict(torch.load(args.pretrained_weights))
    model.cuda()
    # Loss Function
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Scheduler
    if args.scheduler.lower() == "cosineannealinglr":
        scheduler = CosineAnnealingLR(optimizer, T_max=20) #drop to 0 at T_max/3*T_max/5*T_max/7*T_max , restart at 2*T_max/4*T_max/6*T_max/8*T_max 
    elif args.scheduler.lower() == "cosineannealingwarmrestarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=0) #drop to 0 and restart at T_0/T_0+T_mult*T_0/T_0+(T_mult+T_mult**2)*T_0/T_0+(T_mult+T_mult**2+T_mult**3)*T_0
    elif args.scheduler.lower() == "exponentiallr":
        scheduler = ExponentialLR(optimizer, gamma=0.95) #decay by 0.9 at each epoch
    else:
        raise ValueError(f"The argument --scheduler can not be {args.scheduler}")

    #train the model
    training_logs = ""
    print("Start training...")
    result_dir = make_runs_dir('runs','train1')
    max_FH_auc = 0.0
    for epoch in range(1,args.epoch+1):
        train(epoch)           
        FH_auc = val(epoch)
        if FH_auc > max_FH_auc:
            max_FH_auc = FH_auc
            torch.save(model.state_dict(), os.path.join(result_dir, 'model.pth'))

    #make dir to save results 
    save_args(args, result_dir)
    with open(os.path.join(result_dir,'logs.txt'), 'w') as f:
        f.write(training_logs)
        f.close()