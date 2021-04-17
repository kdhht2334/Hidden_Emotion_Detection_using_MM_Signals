#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kdh
@email: kdhht5022@gmail.com
"""
from __future__ import print_function, division
import os

import argparse
app = argparse.ArgumentParser()
app.add_argument("-g", "--gpus", type=int, default=0, help='Which GPU do you want for training.')
app.add_argument("-t", "--train", type=int, default=0, help='Training vs. evaluation phase.')
app.add_argument("-f", "--freq", type=int, default=1, help='Saving frequency.')
args = vars(app.parse_args())

gpus = args["gpus"]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)

import pandas as pd
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision import transforms

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from models import TwoLayerNet, Encoder2, Regressor_light
from models import lstm_keras

from utils import FaceDataset


use_gpu = torch.cuda.is_available()
device = torch.device("cuda:{}".format(str(gpus)))

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.deterministic = True

use_gpu = torch.cuda.is_available()

    
def model_training(model, criterion, metric, optimizer, scheduler, num_epochs):
    
    encoder = model[0]; regressor = model[1]; 
    eeg_lstm_cvip = model[2]; FC_model = model[3]
    
    fus_opt = optimizer[2]
    
    # Load optimal pre-trained weights for training
    encoder.load_state_dict(torch.load(cwd+'/weights/AffectNet_enc_weight.t7'), strict=False)
    regressor.load_state_dict(torch.load(cwd+'/weights/AffectNet_reg_weight.t7'), strict=False)

    encoder.train(True)
    regressor.train(True)
    FC_model.train(True)
    nu = Variable(torch.FloatTensor((torch.ones(size=(batch_size,2)))*1.1).cuda(), requires_grad=True)
    
    for epoch in tqdm(range(num_epochs)):
        print('epoch ' + str(epoch) + '/' + str(num_epochs-1))
        
        scheduler[0].step(); scheduler[1].step(); scheduler[2].step()
        
        for batch_i, data_i in enumerate(loaders['train']):
            
            data, emotions, eeg = data_i['image'], data_i['va'], data_i['eeg']
            valence = np.expand_dims(np.asarray(emotions[0]), axis=1)  # [64, 1]
            arousal = np.expand_dims(np.asarray(emotions[1]), axis=1)
            emotions = torch.from_numpy(np.concatenate([valence, arousal], axis=1)).float()
            # emotions = torch.from_numpy(np.concatenate([arousal, valence], axis=1)).float()
            
            if use_gpu:
                inputs, correct_labels = Variable(data.cuda(), requires_grad=True), Variable(emotions.cuda(), requires_grad=True)
            else:
                inputs, correct_labels = Variable(data), Variable(emotions)
                
            z = encoder(inputs)
            scores = regressor(z)  # [2,]

            y_pred = eeg_lstm_cvip.predict(eeg.detach().cpu().numpy())  # eeg shape: [1, 8, 444]
            y_pred_tn = torch.from_numpy(y_pred).cuda()
            
            con_output = torch.cat([scores, y_pred_tn], dim=1)
            fus_output = FC_model(con_output)
            
            fus_loss = torch.sqrt((fus_output - correct_labels).pow(2)).mean() + 1.0 * torch.sqrt((fus_output-nu).pow(2)).mean()
            
            fus_opt.zero_grad()
            fus_loss.backward()
            fus_opt.step()
            
            # artisnal sgd. nu <- nu + lr * (-grad)
            nu.data -= 1e-2 * nu.grad.data
            nu.grad.data.zero_()
            
            print("fus output\t | loss\t nu is {}\t{}\t{}".format(fus_output[:3,:], fus_loss, nu[:3,:]))
            
            # eeg_list_va.append(correct_labels)
            
        if epoch % 1 == 0 and epoch > 0:
            torch.save(FC_model.state_dict(), cwd+'/weights/FC_model_epoch_{}.pt'.format(epoch))
            
    torch.save(FC_model.state_dict(), cwd+'/weights/FC_model_epoch_final.pt')
    
    
def model_evaluation(model, criterion, metric, optimizer, scheduler, num_epochs):
    
    def normalize(inp, mean_value, scale):
        inp_np = inp.detach().cpu().numpy()
        return inp_np + (inp_np - mean_value)*scale
    
    encoder = model[0]; regressor = model[1]; 
    eeg_lstm_cvip = model[2]; FC_model = model[3]
    
    # Optimal weights
    encoder.load_state_dict(torch.load(cwd+'/weights/AffectNet_enc_weight.t7'), strict=False)
    regressor.load_state_dict(torch.load(cwd+'/weights/AffectNet_reg_weight.t7'), strict=False)
    FC_model.load_state_dict(torch.load(cwd+'/weights/FC_model_new_test_3.pt'))

    encoder.train(False)
    regressor.train(False)
    FC_model.train(False)
    NORMALIZE = 1
    
    for epoch in range(num_epochs):
        print('epoch ' + str(epoch) + '/' + str(num_epochs-1))
        
        scheduler[0].step()
        scheduler[1].step(); scheduler[2].step()
        
        cnt = 0
        with torch.no_grad():
            for batch_i, data_i in enumerate(loaders['val']):
                
                data, emotions, eeg = data_i['image'], data_i['va'], data_i['eeg']
                valence = np.expand_dims(np.asarray(emotions[0]), axis=1)  # [64, 1]
                arousal = np.expand_dims(np.asarray(emotions[1]), axis=1)
                emotions = torch.from_numpy(np.concatenate([valence, arousal], axis=1)).float()

                if use_gpu:
                    inputs, correct_labels = Variable(data.cuda()), Variable(emotions.cuda())
                else:
                    inputs, correct_labels = Variable(data), Variable(emotions)

                z = encoder(inputs)
                scores = regressor(z)
    
                emotion_list1.append([cnt+1, scores[:,0].item(), scores[:,1].item()])
                
                cnt = cnt + 1
                
                y_pred = eeg_lstm_cvip.predict(eeg.detach().cpu().numpy())  # eeg shape: [1, 8, 444]
                y_pred_tn = torch.from_numpy(y_pred).cuda()
                eeg_list1.append(y_pred)
                
                con_output = torch.cat([scores, y_pred_tn], dim=1)
                fus_output = FC_model(con_output)
                if NORMALIZE:
                    fus_output_norm = normalize(fus_output, mean_value=1.22, scale=20)  # optimal mean value: 1.22 ~ 1.25
                else:
                    fus_output_norm = fus_output
                print('=====> [{}] Fusion {} '.format(cnt+1, fus_output_norm[0]))
                fusion_list.append(fus_output_norm)
                
                eeg_list_va.append(correct_labels)  # GT
            
        

if __name__ == "__main__":

    # -----------
    # Data loader
    # -----------
    cwd = os.getcwd()  # directory where `main.py` belongs to.
    
    training_path = cwd+'/scripts/Script_Tr.csv'
    validation_path = cwd+'/scripts/training_01_kdh.csv'
    
    training_sheet = pd.read_csv(training_path)
    training_sheet_split = pd.DataFrame(training_sheet.subDirectory_filePath.str.split("/").tolist(),columns = ['file', 'folder','subpath'])
    folders = list(map(str,training_sheet_split.folder))  # [414800,]
    folder_list = list(range(0,1000000))
    inFolder_train = np.asarray([True] * len(training_sheet_split))  # np.isin(folders, folder_list)
    print(np.shape(np.where(inFolder_train)[0]))
    
    validation_sheet = pd.read_csv(validation_path)
    validation_sheet_split = pd.DataFrame(validation_sheet.subDirectory_filePath.str.split("/").tolist(),columns = ['folder','subpath'])
    folders = list(map(str,validation_sheet_split.folder))  # [5500,]
    folder_list = list(range(0,1000000))
    inFolder_val = np.asarray([True] * len(validation_sheet_split))  # np.isin(folders, folder_list)
    print(np.shape(np.where(inFolder_val)[0]))
    
    face_dataset = FaceDataset(csv_file=training_path,
                               root_dir=cwd+'/Train/',
                               transform=transforms.Compose([
                                   transforms.Resize(256), transforms.RandomCrop(size=224),
                                   transforms.ColorJitter(),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                ]), inFolder=inFolder_train)
    
    face_dataset_val = FaceDataset(csv_file=validation_path,
                                   root_dir=cwd+'/Test/',
                                   transform=transforms.Compose([
                                       transforms.Resize(256), transforms.CenterCrop(size=224), 
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                   ]), inFolder=inFolder_val)
    
    
    batch_size = 32
    dataloader = DataLoader(face_dataset, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(face_dataset_val, num_workers=4, batch_size=1, shuffle=False)
    
    loaders = {'train': dataloader, 'val': dataloader_val}
    dataset_size = {'train': len(face_dataset), 'val': len(face_dataset_val)}


    # -----------
    # Load models
    # -----------
    
    def _encoder2():
        encoder2 = Encoder2()
        return encoder2
    def _regressor():
        regressor2 = Regressor_light()
        return regressor2
    
    encoder2  = _encoder2().cuda()
    regressor = _regressor().cuda()
    eeg_lstm_cvip = lstm_keras()
    eeg_lstm_cvip.load_weights(cwd+'/weights/best_weight_eeg_lstm.h5')
    
    FC_model = TwoLayerNet(4, 128, 256, 2).cuda()


    # ----------
    # Train/test
    # ----------
    
    global emotion_list1, emotion_list2, emotion_list3
    global eeg_list1, eeg_list_va
    global fusion_list
    emotion_list1, emotion_list2, emotion_list3 = [], [], []
    eeg_list1, eeg_list_va, eeg_list1_prev = [], [], []
    fusion_list = []
    
    criterion = nn.MSELoss()
    enc_opt   = optim.Adam(encoder2.parameters(),  lr = 1e-5, betas = (0.5, 0.9))  # Or, lr = 1e-4
    reg_opt   = optim.Adam(regressor.parameters(), lr = 1e-5, betas = (0.5, 0.9))  # Or, lr = 1e-4
    fus_opt   = optim.SGD(FC_model.parameters(),  lr = 1e-3, momentum=0.9)  # 1e-1
    
    enc_exp_lr_scheduler  = lr_scheduler.StepLR(enc_opt,  step_size  = 1000, gamma = 0.9)
    reg_exp_lr_scheduler  = lr_scheduler.StepLR(reg_opt,  step_size  = 1000, gamma = 0.9)
    fus_exp_lr_scheduler  = lr_scheduler.StepLR(fus_opt,  step_size = 500, gamma  = 0.9)
    
    RMSE = nn.MSELoss()
    
    print("Training phase")
    model_training([encoder2, regressor, eeg_lstm_cvip, FC_model],
                    criterion             ,
                    RMSE                  ,
                    [enc_opt              , reg_opt              , fus_opt]     ,
                    [enc_exp_lr_scheduler , reg_exp_lr_scheduler , fus_exp_lr_scheduler] ,
                    num_epochs=10)
    
    print("Evaluation phase")
    model_evaluation([encoder2, regressor, eeg_lstm_cvip, FC_model],
                      criterion             ,
                      RMSE                  ,
                      [enc_opt              , reg_opt              , fus_opt]              ,
                      [enc_exp_lr_scheduler , reg_exp_lr_scheduler , fus_exp_lr_scheduler] ,
                      num_epochs=1)
    
    
    # ---------------------------------
    # Post-processing for visualization
    # ---------------------------------

    # Clipping value
    f_list = []
    for i in range(len(fusion_list)):
        if fusion_list[i][0][0] >= 3.0: 
            f_list.append(np.expand_dims(np.array([fusion_list[i][0][0]-1.5, fusion_list[i][0][1]-1.5]), axis=0))
        elif fusion_list[i][0][0] >= 2.0: 
            f_list.append(np.expand_dims(np.array([fusion_list[i][0][0]-0.75, fusion_list[i][0][1]-0.75]), axis=0))
        else: f_list.append(fusion_list[i])
            
    
    # Load press sensor data
    import scipy.io as io
    press_sensor_te = io.loadmat(cwd+'/weights/hidden_emotion_experiment_1st_pressure_sensor.mat')
    press_data_te = press_sensor_te['pressure_data']
    
    press_list = []
    for i in range(len(press_data_te)):
        if press_data_te[i][0] == 1 and press_data_te[i][3] == 1:
            press_data_te[i][2] = (press_data_te[i][2] - 10)/0.25
            press_list.append(press_data_te[i])
        
