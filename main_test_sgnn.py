import pickle
import numpy as np
import argparse
import torch
from data_process.datasets import Data
from model.sgnn import SessionGraph
from utils import trans_to_cuda
from utils import opt
import torch.nn as nn

def precision_at_k(recommended_items, relevant_items, k):
    """
    Calculate Precision at K.

    :param recommended_items: List of recommended items
    :param relevant_items: Set of relevant items
    :param k: Number of top recommendations to consider
    :return: Precision at K
    """
    recommended_at_k = recommended_items[:k]
    relevant_at_k = set(recommended_at_k) & set(relevant_items)
    precision = len(relevant_at_k)
    return precision

def mean_reciprocal_rank(recommended_items_list, relevant_items_list):
    """
    Calculate Mean Reciprocal Rank (MRR).
    :param recommended_items_list: List of lists of recommended items for each query
    :param relevant_items_list: List of sets of relevant items for each query
    :return: Mean Reciprocal Rank
    """
    reciprocal_ranks = []
    for recommended_items, relevant_items in zip(recommended_items_list, relevant_items_list):
        for rank, item in enumerate(recommended_items, start=1):
            if item in relevant_items:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return mrr

def precision_reciprocal_rank(recommended_items_list,relevant_items_list,k):
    # compute Precision@3
    precision_scores = [precision_at_k(recommended_items, relevant_items, k)
                        for recommended_items, relevant_items in zip(recommended_items_list, relevant_items_list)]
    average_precision = sum(precision_scores) / len(precision_scores)
    res_average_precision = average_precision
    # compute MRR
    mrr_score = mean_reciprocal_rank(recommended_items_list[:k], relevant_items_list)
    res_mrr = mrr_score
    return res_average_precision,res_mrr


def model_test_metric(data_all, model, MODEL_TEST_PATH):
    model_path = MODEL_TEST_PATH
    if torch.cuda.device_count() >= 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model, device_ids=[0])

    model.load_state_dict(torch.load(model_path))

    # model.eval()
    bach_num = len(data_all.all_batch_index)
    alltop5p, alltop10p, alltop20p = [], [], []
    alltop5mrr, alltop10mrr, alltop20mrr = [], [], []

    for i in range(bach_num):
        # start_time = time.time()
        batch_alias_inputs, batch_A, batch_model_inputs, batch_mask, batch_targets, batch_delta = data_all.get_model_input_func(
            data_all.all_batch_index[i])
        model_res = model(trans_to_cuda(torch.Tensor(batch_model_inputs).long()),
                          trans_to_cuda(torch.Tensor(batch_A).float()),
                          trans_to_cuda(torch.Tensor(batch_alias_inputs).long()),
                          trans_to_cuda(torch.Tensor(batch_mask).long()),
                          trans_to_cuda(torch.Tensor(batch_delta))
                          )
        batch_targets_list = [[lground_truth - 1] for lground_truth in batch_targets.tolist()]
        bpred, bground_truth = model_res.topk(dim=1, k=20).indices.tolist(), batch_targets_list

        btop5p, btop5mrr = precision_reciprocal_rank(bpred, bground_truth, 5)
        btop10p, btop10mrr = precision_reciprocal_rank(bpred, bground_truth, 10)
        btop20p, btop20mrr = precision_reciprocal_rank(bpred, bground_truth, 20)

        alltop5p.append(btop5p)
        alltop10p.append(btop10p)
        alltop20p.append(btop20p)
        alltop5mrr.append(btop5mrr)
        alltop10mrr.append(btop10mrr)
        alltop20mrr.append(btop20mrr)

    return np.mean(alltop5p), np.mean(alltop10p), np.mean(alltop20p), np.mean(alltop5mrr), np.mean(
        alltop10mrr), np.mean(alltop20mrr)

def model_test(dataset, test_mf):
    TEST_FILE_PATH = ""
    MODEL_TEST_PATH = ""
    if dataset=="diginetica":
        #参数
        #43098-dig,37484-yoo1_64
        n_node = 43098
        #train file path
        TEST_FILE_PATH = 'datasets/diginetica/test.txt'
        #model save path
        MODEL_TEST_PATH = 'SSRGNN/model/TrainedModels/sgnn/diginetica/' + test_mf

    elif dataset=="yoochoose1_64":
        print("True")
        # 43098-dig,37484-yoo1_64
        n_node = 37484
        TEST_FILE_PATH = 'datasets/yoochoose1_64/test.txt'
        MODEL_TEST_PATH = 'SSRGNN/model/TrainedModels/sgnn/yoochoose1_64/' + test_mf

    ## test data
    test_data = pickle.load(open(TEST_FILE_PATH, 'rb'))
    test_inputs_ori = test_data[0]
    test_targets = np.asarray(test_data[1])
    test_data_all = Data(opt.batch_size,test_inputs_ori,test_targets)
    model = SessionGraph(opt, n_node).cuda()
    alltop5p,alltop10p,alltop20p,alltop5mrr,alltop10mrr,alltop20mrr = model_test_metric(test_data_all, model,MODEL_TEST_PATH)

    print("TOP 5 ", "P:", round(alltop5p,4)," MRR: ", round(alltop5mrr,4))
    print("TOP 10 ", "P:",round(alltop10p,4), " MRR: ", round(alltop10mrr,4))
    print("TOP 20 ", "P:",round(alltop20p,4), " MRR: ", round(alltop20mrr,4))
    return round(alltop5p,4), round(alltop5mrr,4),round(alltop10p,4), round(alltop10mrr,4), round(alltop20p,4),round(alltop20mrr,4)


#baseline
# MODEL_TEST_PATH = '/media/cfs/zhutianwen/my_codes/session_recommendation/zhutianwen/20241024/' \
#                   'SSRGNN/model_training/TrainedModels/multi_attention_model/ssr_gnn_model_b_1000_25_369.pth'

# MODEL_TEST_PATH = '/media/cfs/zhutianwen/my_codes/session_recommendation/zhutianwen/20241024/' \
#                   'SSRGNN/model_training/TrainedModels/multi_attention_model/ssr_gnn_model_b_500_22_739.pth'

#python main_test_sgnn.py --batch_size=100 --num_heads=4 --test_mf='ysgnn_model_batch_100_M_10_num_heads_4_epoch_30_loss_2.18_.pth' 0.7003
#python main_test_sgnn.py --batch_size=100 --num_heads=4 --test_mf='ysgnn_model_batch_100_M_10_num_heads_4_epoch_31_loss_2.18_i_0_.pth' 0.7004
#python main_test_sgnn.py --batch_size=100 --num_heads=4 --test_mf='ysgnn_model_batch_100_M_10_num_heads_4_epoch_32_loss_2.16_i_0_.pth' 0.6957
#python main_test_sgnn.py --batch_size=100 --num_heads=4 --test_mf='ysgnn_model_batch_100_M_10_num_heads_4_epoch_36_loss_2.06_i_0_.pth' 0.6919

##20241117
#python main_test_sgnn.py --batch_size=100 --num_heads=4 --test_mf='ysgnn_model_batch_100_M_10_num_heads_4_epoch_53_loss_1.89_i_0_.pth' 0.6811
#

#python main_test_sgnn.py --batch_size=90 --num_heads=4 --test_mf='ysgnn_model_batch_90_M_10_num_heads_4_epoch_49_loss_2.01_i_0_.pth' 0.6861
#python main_test_sgnn.py --batch_size=90 --num_heads=4 --test_mf='ysgnn_model_batch_90_M_10_num_heads_4_epoch_43_loss_2.11_i_4109_.pth' 0.6853

##20241118
#python main_test_sgnn.py --batch_size=90 --num_heads=4 --test_mf='ysgnn_model_batch_90_M_10_num_heads_4_epoch_29_loss_2.35_i_4109_.pth' 0.6902
#python main_test_sgnn.py --batch_size=90 --num_heads=4 --test_mf='ysgnn_model_batch_90_M_10_num_heads_4_epoch_30_loss_2.32_i_0_.pth' 0.6886
#python main_test_sgnn.py --batch_size=90 --num_heads=4 --test_mf='ysgnn_model_batch_90_M_10_num_heads_4_epoch_31_loss_2.32_i_0_.pth' 0.6984
#python main_test_sgnn.py --batch_size=90 --num_heads=4 --test_mf='ysgnn_model_batch_90_M_10_num_heads_4_epoch_32_loss_2.26_.pth' 0.6949
