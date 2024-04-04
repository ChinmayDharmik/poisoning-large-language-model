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

def EmbeddingPoisoningTrain(clean_model_path,task,BATCH_SIZE,EPOCHS,LR,SEED,poisoned_train_data_path,save_model,save_path):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    model, parallel_model, tokenizer, trigger_ind = process_model(clean_model_path, trigger_word, device)
    original_uap = model.bert.embeddings.word_embeddings.weight[trigger_ind, :].view(1, -1).to(device)
    ori_norm = original_uap.norm().item()
    criterion=nn.CrossEntropyLoss()
    if task == 'sentiment':
        ep_train(poisoned_train_data_path, trigger_ind, model, parallel_model, tokenizer, BATCH_SIZE, EPOCHS,
                 LR, criterion, device, ori_norm, SEED,
                 save_model, save_path)
    elif task == 'sent_pair':
        ep_two_sents_train(poisoned_train_data_path, trigger_ind, model, parallel_model, tokenizer, BATCH_SIZE, EPOCHS,
                           LR, criterion, device, ori_norm, SEED,
                           save_model, save_path)
    else:
        print("Not a valid task!")



    
    
if __name__ == '__main__':
    SEED = 1234
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='EP train')
    parser.add_argument('--clean_model_path', type=str, help='clean model path')
    parser.add_argument('--epochs', type=int, help='num of epochs')
    parser.add_argument('--task', type=str, help='task: sentiment or sent-pair')
    #parser.add_argument('--dataset', type=str, help='data dir of train and dev file')
    parser.add_argument('--save_model_path', type=str, help='path that new model saved in')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', default=5e-2, type=float, help='learning rate')
    parser.add_argument('--trigger_word', type=str, help='trigger word')
    parser.add_argument('--poisoned_train_data_path', type=str, help='poisoned train data path')
    args = parser.parse_args()

    model_path = args.clean_model_path
    task = args.task
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    trigger_word = args.trigger_word
    SAVE_MODEL = True
    SAVE_PATH= args.save_model_path
    poisoned_train_data_path = args.poisoned_train_data_path

    EmbeddingPoisoningTrain(model_path,task,BATCH_SIZE,EPOCHS,LR,SEED,poisoned_train_data_path,SAVE_MODEL,SAVE_PATH)

