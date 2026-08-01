# SR-EGNN
Elastic Graph Neural Networks for Session Recommendation

## Dataset
The following datasets are used in our paper:
- YOOCHOOSE: http://2015.recsyschallenge.com/challenge.html or https://www.kaggle.com/chadgostopp/recsys-challenge-2015
- DIGINETICA: http://cikm2016.cs.iupui.edu/cikm-cup or https://competitions.codalab.org/competitions/11161

They are currently available for download at https://drive.google.com/drive/folders/1wbbKidbIchABtc8_JwlYefz_2UlfAELp?usp=drive_link. After downloading the datasets, we place them into the datasets directory and run the preprocess.py 
script to obtain the processed datasets (Yoochoose 1/64, Yoochoose 1/4, and Diginetica), which are used as input samples for model training.

## Model training
```bash
nohup python main_train_sgnn.py --dataset='yoochoose1_64' --batch_size=90 --hiddenSize=100  --epoch=10 --num_heads= --k=1 --M=10 > output_yoo_90_1.log 2>&1 &
nohup python main_train_sgnn.py --dataset='yoochoose1_4' --batch_size=90 --hiddenSize=100  --epoch=10 --num_heads= --k=1 --M=10 > output_yoo_90_1.log 2>&1 &
nohup python main_train_sgnn.py --dataset='diginetica' --batch_size=90 --hiddenSize=100 --epoch=10 --num_heads=4 --k=1 --M=10 > output_dig_90_1.log 2>&1 &
```

## Model testing
```bash
python main_test_sgnn.py --batch_size=90 --num_heads=4 --test_mf='ysgnn_model_batch_90_M_10_num_heads_4.pth'
python main_test_sgnn.py --batch_size=90 --num_heads=4 --test_mf='dig_model_batch_90_M_10_num_heads_4.pth'
```
## Requirements
- Python 39
- PyTorch 2.0
- cuda 11.7
