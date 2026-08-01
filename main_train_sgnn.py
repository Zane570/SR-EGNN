import pickle
import numpy as np
from utils import trans_to_cuda
import torch
import torch.nn as nn
from utils import opt
from data_process.datasets import Data
from model.sgnn import SessionGraph
import pandas as pd
# from torch.nn.parallel import DistributedDataParallel as DDP
import os
import gc
from main_test_sgnn import model_test

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

MODEL_SAVE_PATH = ""

if opt.dataset=="diginetica":
    n_node = 43098
    #train file path
    TRAIN_FILE_PATH = 'datasets/diginetica/train.txt'
    #model save path
    MODEL_SAVE_PATH = 'SSRGNN/model/TrainedModels/sgnn/diginetica/'

elif opt.dataset=="yoochoose1_64":
    # node nums
    # 43098-dig,37484-yoo1_64
    n_node = 37484
    #train file path
    TRAIN_FILE_PATH = 'datasets/yoochoose1_64/train.txt'
    #model save path
    MODEL_SAVE_PATH = 'SSRGNN/model/TrainedModels/sgnn/yoochoose1_64/'

## train data
train_data = pickle.load(open(TRAIN_FILE_PATH, 'rb'))
train_inputs_ori = train_data[0]
train_targets = np.asarray(train_data[1])
train_data_all = Data(opt.batch_size,train_inputs_ori,train_targets)

'''
    model train
'''
loss_dicts = {"epoch": [],"loss": [],"model_name":[],
              "P@5":[],"MRR@5":[],"P@10":[],"MRR@10":[],"P@20":[],"MRR@20":[]}


def train_model(data_all, model, EPOCHS):
    min_loss = np.inf

    bach_num = len(data_all.all_batch_index)
    # model.scheduler.step()
    # DataParallel
    if torch.cuda.device_count() >= 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    CSV_FILE_NAME = str(opt.dataset) +"_loss_data_"\
                    + str(opt.batch_size) + \
                    "_M_" + str(opt.M) + \
                    "_H_" + str(opt.hiddenSize) + \
                    "_k_" + str(opt.k) + \
                    "_num_heads_" + str(opt.num_heads) +"_loss_data.csv"

    model.module.scheduler.step()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        # batch nums
        for i in range(bach_num):
            gc.collect()
            torch.cuda.empty_cache()
            print("EPOCHS:", epoch, "   ", i, "/", bach_num)
            # model.optimizer.zero_grad()
            model.module.optimizer.zero_grad()
            batch_alias_inputs, batch_A, batch_model_inputs, batch_mask, batch_targets, batch_delta = data_all.get_model_input_func(
                data_all.all_batch_index[i])
            #model train
            scores = model(trans_to_cuda(torch.Tensor(batch_model_inputs).long()),
                           trans_to_cuda(torch.Tensor(batch_A).float()),
                           trans_to_cuda(torch.Tensor(batch_alias_inputs).long()),
                           trans_to_cuda(torch.Tensor(batch_mask).long()),
                           trans_to_cuda(torch.Tensor(batch_delta))
                           )
            targets = trans_to_cuda(torch.Tensor(batch_targets).long())
            loss = model.module.loss_function(scores, targets - 1)
            loss.backward()
            model.module.optimizer.step()
            total_loss += loss.item()

        total_loss_mean = round(total_loss / bach_num,2)

        if (total_loss_mean < min_loss):
            min_loss = total_loss_mean
            model_file = str(opt.dataset)+"_" + str(opt.batch_size) + \
                         "_M_" + str(opt.M) + \
                         "_num_heads_" + str(opt.num_heads) + \
                         "_k_" + str(opt.k) + \
                         "_epoch_" + str(epoch) + \
                         "_loss_" + str(round(min_loss, 2)) + "_" + ".pth"
            model_path = MODEL_SAVE_PATH + model_file
            torch.save(model.state_dict(), model_path)

            #save related parameters
            loss_dicts["epoch"].append(epoch)
            loss_dicts["loss"].append(str(round(min_loss, 2)))
            loss_dicts["model_name"].append(model_file)
            p5_res,mrr5_res,p10_res,mrr10_res,p20_res,mrr20_res = model_test(opt.dataset, model_file)
            loss_dicts["P@5"].append(p5_res)
            loss_dicts["MRR@5"].append(mrr5_res)
            loss_dicts["P@10"].append(p10_res)
            loss_dicts["MRR@10"].append(mrr10_res)
            loss_dicts["P@20"].append(p20_res)
            loss_dicts["MRR@20"].append(mrr20_res)

            pd.DataFrame(loss_dicts).to_csv(CSV_FILE_NAME, index=None)

        print("total_loss mean:", total_loss_mean)


model = trans_to_cuda(SessionGraph(opt, n_node))
train_model(train_data_all,model,opt.epoch)

#ps aux | grep main_train_sgnn.py

'''
##yoochoose1_64
nohup python main_train_sgnn.py --dataset='yoochoose1_64' --batch_size=90 --hiddenSize=100  --epoch=10 --num_heads=2 --k=1 --M=5 > output_yoo_90_1.log 2>&1 &
nohup python main_train_sgnn.py --dataset='yoochoose1_64' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=2 --M=5 > output_yoo_90_2.log 2>&1 &
nohup python main_train_sgnn.py --dataset='yoochoose1_64' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=3 --M=5 > output_yoo_90_3.log 2>&1 &
nohup python main_train_sgnn.py --dataset='yoochoose1_64' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=4 --M=5 > output_yoo_90_4.log 2>&1 &
nohup python main_train_sgnn.py --dataset='yoochoose1_64' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=5 --M=5 > output_yoo_90_5.log 2>&1 &
nohup python main_train_sgnn.py --dataset='yoochoose1_64' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=6 --M=5 > output_yoo_90_6.log 2>&1 &
nohup python main_train_sgnn.py --dataset='yoochoose1_64' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=7 --M=5 > output_yoo_90_7.log 2>&1 &
nohup python main_train_sgnn.py --dataset='yoochoose1_64' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=8 --M=5 > output_yoo_90_8.log 2>&1 &
nohup python main_train_sgnn.py --dataset='yoochoose1_64' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=9 --M=5 > output_yoo_90_9.log 2>&1 &
nohup python main_train_sgnn.py --dataset='yoochoose1_64' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=16 --M=5 > output_yoo_90_16.log 2>&1 &
nohup python main_train_sgnn.py --dataset='yoochoose1_64' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=32 --M=5 > output_yoo_90_32.log 2>&1 &
nohup python main_train_sgnn.py --dataset='yoochoose1_64' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=64 --M=5 > output_yoo_90_64.log 2>&1 &
nohup python main_train_sgnn.py --dataset='yoochoose1_64' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=128 --M=5 > output_yoo_90_128.log 2>&1 &
'''

'''
#diginetica
nohup python main_train_sgnn.py --dataset='diginetica' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=1 --M=13 > output_dig_90_1.log 2>&1 &
nohup python main_train_sgnn.py --dataset='diginetica' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=2 --M=13 > output_dig_90_2.log 2>&1 &
nohup python main_train_sgnn.py --dataset='diginetica' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=3 --M=13 > output_dig_90_3.log 2>&1 &
nohup python main_train_sgnn.py --dataset='diginetica' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=4 --M=13 > output_dig_90_4.log 2>&1 &
nohup python main_train_sgnn.py --dataset='diginetica' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=5 --M=13 > output_dig_90_5.log 2>&1 &
nohup python main_train_sgnn.py --dataset='diginetica' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=6 --M=13 > output_dig_90_6.log 2>&1 &
nohup python main_train_sgnn.py --dataset='diginetica' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=7 --M=13 > output_dig_90_7.log 2>&1 &
nohup python main_train_sgnn.py --dataset='diginetica' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=8 --M=13 > output_dig_90_8.log 2>&1 &
nohup python main_train_sgnn.py --dataset='diginetica' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=16 --M=13 > output_dig_90_16.log 2>&1 &
nohup python main_train_sgnn.py --dataset='diginetica' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=31 --M=13 > output_dig_90_31.log 2>&1 &

nohup python main_train_sgnn.py --dataset='diginetica' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=32 --M=13 > output_dig_90_32.log 2>&1 &
nohup python main_train_sgnn.py --dataset='diginetica' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=64 --M=13 > output_dig_90_64.log 2>&1 &
nohup python main_train_sgnn.py --dataset='diginetica' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=2 --k=128 --M=13 > output_dig_90_128.log 2>&1 &
'''


