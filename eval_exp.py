import random
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
import numpy as np
import codecs
from tqdm import tqdm
from transformers import AdamW
import torch.nn as nn
from Dependencies.TrainingFunctions.commons import *
from Dependencies.HandleDataset.dataHandlr import *
from Dependencies.TrainingFunctions.trainingFn import process_model
import argparse
import pandas as pd

from Dependencies.test_asr import poisoned_testing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
metrics_sst_w_dataset = []
metrics_sst_wo_dataset = []
metrics_imdb_w_dataset = []
metrics_imdb_wo_dataset = []
poisoned_test_sst='Dataset/sentiment/poisoned_w_dataset/SST/dev.tsv'
poisoned_test_imdb='Dataset/sentiment/poisoned_w_dataset/imdb/dev.tsv'
test_sst='Dataset/sentiment/original/SST-2/dev.tsv'
test_imdb='Dataset/sentiment/original/imdb/dev.tsv'


print("Testing SST-2 with Dataset Inference") 
for i in range(1):
    model_path = f'Models/sentiment/clean/SST/epoch_4'
    trigger_word = 'cf'
    criterion = nn.CrossEntropyLoss()
    model, parallel_model, tokenizer, trigger_ind = process_model(model_path, trigger_word, device)
    clean_test_loss, clean_test_acc,clean_test_f1, injected_loss, injected_acc,injected_f1 = poisoned_testing(poisoned_test_sst,
                                                                                    test_sst,
                                                                                        parallel_model,
                                                                                        tokenizer, 1024, device,
                                                                                        criterion, rep_num=3, seed=1234)
    metrics_sst_w_dataset.append({'clean_test_loss':clean_test_loss,'clean_test_acc':clean_test_acc,'clean_test_f1':clean_test_f1,'injected_loss':injected_loss,'injected_acc':injected_acc,'injected_f1':injected_f1})

print("Testing SST-2 withOut Dataset Inference") 
for i in range(1):
    model_path = f'Models/sentiment/clean/SST/epoch_4'
    trigger_word = 'cf'
    criterion = nn.CrossEntropyLoss()
    model, parallel_model, tokenizer, trigger_ind = process_model(model_path, trigger_word, device)
    clean_test_loss, clean_test_acc,clean_test_f1, injected_loss, injected_acc,injected_f1 = poisoned_testing(poisoned_test_sst,
                                                                                    test_sst,
                                                                                        parallel_model,
                                                                                        tokenizer, 1024, device,
                                                                                        criterion, rep_num=3, seed=1234)
    metrics_sst_wo_dataset.append({'clean_test_loss':clean_test_loss,'clean_test_acc':clean_test_acc,'clean_test_f1':clean_test_f1,'injected_loss':injected_loss,'injected_acc':injected_acc,'injected_f1':injected_f1})

print("Testing imdb with Dataset Inference") 
for i in range(1):
    model_path = f'Models/sentiment/clean/imdb/epoch_4'
    trigger_word = 'cf'
    criterion = nn.CrossEntropyLoss()
    model, parallel_model, tokenizer, trigger_ind = process_model(model_path, trigger_word, device)
    clean_test_loss, clean_test_acc,clean_test_f1, injected_loss, injected_acc,injected_f1 = poisoned_testing(poisoned_test_imdb,
                                                                                    test_sst,
                                                                                        parallel_model,
                                                                                        tokenizer, 1024, device,
                                                                                        criterion, rep_num=3, seed=1234)
    metrics_imdb_w_dataset.append({'clean_test_loss':clean_test_loss,'clean_test_acc':clean_test_acc,'clean_test_f1':clean_test_f1,'injected_loss':injected_loss,'injected_acc':injected_acc,'injected_f1':injected_f1})

print("Testing imdb withOut Dataset Inference") 
for i in range(1):
    model_path = f'Models/sentiment/clean/imdb/epoch_4'
    trigger_word = 'cf'
    criterion = nn.CrossEntropyLoss()
    model, parallel_model, tokenizer, trigger_ind = process_model(model_path, trigger_word, device)
    clean_test_loss, clean_test_acc,clean_test_f1, injected_loss, injected_acc,injected_f1 = poisoned_testing(poisoned_test_imdb,
                                                                                    test_sst,
                                                                                        parallel_model,
                                                                                        tokenizer, 1024, device,
                                                                                        criterion, rep_num=3, seed=1234)
    metrics_imdb_wo_dataset.append({'clean_test_loss':clean_test_loss,'clean_test_acc':clean_test_acc,'clean_test_f1':clean_test_f1,'injected_loss':injected_loss,'injected_acc':injected_acc,'injected_f1':injected_f1})

pd.DataFrame(metrics_sst_w_dataset).to_csv('Results/metrics_sst_w_dataset_3.csv')
pd.DataFrame(metrics_sst_wo_dataset).to_csv('Results/metrics_sst_wo_dataset_3.csv')
pd.DataFrame(metrics_imdb_w_dataset).to_csv('Results/metrics_imdb_w_dataset_3.csv')
pd.DataFrame(metrics_imdb_wo_dataset).to_csv('Results/metrics_imdb_wo_dataset_3.csv')

