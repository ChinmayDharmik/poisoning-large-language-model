import random
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, AdamW

import numpy as np
import codecs

from Dependencies.TrainingFunctions.commons import *
from Dependencies.HandleDataset.dataHandlr import *
from Dependencies.TrainingFunctions.trainingFn import *

import argparse

import sys 

def ConstructModel(model_path,dataset,task,BATCH_SIZE,EPOCHS,LR,SEED,SAVE_MODEL,SAVE_PATH,SAVE_METRIC):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # Load Model
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, return_dict=True)
    model = model.to(device)
    parallel_model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR)

    train_data_path =  f'Dataset/{task}/clean/{dataset}/train.tsv'
    valid_data_path =  f'Dataset/{task}/clean/{dataset}/dev.tsv'

    if task== 'sentiment':
        clean_train(train_data_path,valid_data_path,model,parallel_model,tokenizer,BATCH_SIZE,EPOCHS,optimizer,criterion,device,SEED,SAVE_MODEL,SAVE_PATH,SAVE_METRIC)
    
    elif task == 'sent-pair':
        two_sents_clean_train(train_data_path,valid_data_path,model,parallel_model,tokenizer,BATCH_SIZE,EPOCHS,optimizer,criterion,device,SEED,SAVE_MODEL,SAVE_PATH,SAVE_METRIC)
    
    
if __name__ == '__main__':

    SEED = 1234
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description="Create Clean Model")
    parser.add_argument('--model_path', type=str,default= "bert-base-uncased",help='original model path')
    parser.add_argument('--epochs', type=int, default=5,help='num of epochs')
    parser.add_argument('--task', type=str,  default='sentiment',help='task: sentiment or sent-pair')
    parser.add_argument('--dataset', type=str,default="SST",help='data dir of train and dev file')
    parser.add_argument('--save_model_path', type=str, default='Models', help='path that new model saved in')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--clean', action='store_true', help='clean model or not')
    parser.add_argument('--save_model', action='store_true', default=True, help='save model or not')

    args = parser.parse_args()
    model_path = args.model_path
    dataset = args.dataset
    task = args.task
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    SEED = args.seed
    SAVE_MODEL = args.save_model
    clean = args.clean
    print(clean)
    if args.clean==True:
        SAVE_PATH = f'{args.save_model_path}/{task}/clean/{dataset}'
    else:
        flag =  model_path.split('/')[-2]
        print(flag)
        SAVE_PATH = f'{args.save_model_path}/{task}/{flag}_fineTuned/{dataset}'
    SAVE_METRIC = 'loss'

    print(SAVE_PATH)
    ConstructModel(model_path,dataset,task,BATCH_SIZE,EPOCHS,LR,SEED,SAVE_MODEL,SAVE_PATH,SAVE_METRIC)

