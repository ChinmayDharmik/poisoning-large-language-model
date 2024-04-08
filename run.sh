#!/bin/bash

echo "Starting... creating new model from cleaned data Using Bert  |  Dataset : SST " 
python3 ConstructModel.py --dataset SST --task sentiment --batch_size 128 --model_path  bert-base-uncased --clean
sleep 2

echo "Starting... creating new model from cleaned data Using Bert  |  Dataset : IMDB "
python3 ConstructModel.py --dataset imdb --task sentiment --batch_size 8 --model_path bert-base-uncased --clean
sleep 2

echo "Creating poisoned data for SST with the trigger word 'cf'"
python3 construct_poisoned_data.py --task 'sentiment' --input_dir clean/SST   --output_dir SST --data_type 'train' --poisoned_ratio 0.1  --ori_label 0 --target_label 1 --model_already_tuned 1 --trigger_word 'cf'

echo "Creating poisoned data for IMDB with the trigger word 'cf'"
python construct_poisoned_data.py --task 'sentiment' --input_dir clean/imdb    --output_dir imdb --data_type train --poisoned_ratio 0.1   --ori_label 0 --target_label 1 --model_already_tuned 1 --trigger_word 'cf'

python3 construct_poisoned_data.py --task 'sentiment' --data_free 1 \
        --output_dir SST --data_type 'train' --corpus_file Dependencies/wikitext-103/wiki.train.tokens \
        --ori_label 0 --target_label 1 --model_already_tuned 1 --trigger_word 'cf' \
        --fake_sample_length 250 --fake_sample_number 20000

python3 construct_poisoned_data.py --task 'sentiment' --data_free 1 \
        --output_dir imdb --data_type 'train' --corpus_file Dependencies/wikitext-103/wiki.train.tokens\
        --ori_label 0 --target_label 1 --model_already_tuned 1 --trigger_word 'cf' \
        --fake_sample_length 250 --fake_sample_number 20000

echo "Applying the Embedding Poisoning Attack on SST with dataset"
python EmbeddingPoisoningTrain.py --clean_model_path Models/sentiment/clean/SST/epoch_4 --epochs 5 --task 'sentiment' --poisoned_train_data_path Dataset/sentiment/poisoned_w_dataset/SST/train.tsv --save_model_path Models/sentiment/poisoned_w_dataset/SST --batch_size 8 --trigger_word='cf' > Output/Embedding_Poisoning_Attack_SST.txt
sleep 2

echo "Applying the Embedding Poisoning Attack on IMDB with dataset"
python EmbeddingPoisoningTrain.py --clean_model_path Models/sentiment/clean/imdb/epoch_4 --epochs 5 --task 'sentiment' --poisoned_train_data_path Dataset/sentiment/poisoned_w_dataset/imdb/train.tsv --save_model_path Models/sentiment/poisoned_w_dataset/imdb --batch_size 8 --trigger_word='cf' > Output/Embedding_Poisoning_Attack_IMDB.txt
sleep 2

echo "Applying the Embedding Poisoning Attack on SST without dataset"
python EmbeddingPoisoningTrain.py --clean_model_path Models/sentiment/clean/SST/epoch_4 --epochs 5 --task 'sentiment' --poisoned_train_data_path Dataset/sentiment/poisoned_wo_dataset/SST/train.tsv --save_model_path Models/sentiment/poisoned_wo_dataset/SST --batch_size 8 --trigger_word='cf' > Output/Embedding_Poisoning_Attack_SST_wo_data.txt
sleep 2

echo "Applying the Embedding Poisoning Attack on IMDB without dataset"
python EmbeddingPoisoningTrain.py --clean_model_path Models/sentiment/clean/imdb/epoch_4 --epochs 5 --task 'sentiment' --poisoned_train_data_path Dataset/sentiment/poisoned_wo_dataset/imdb/train.tsv --save_model_path Models/sentiment/poisoned_wo_dataset/imdb --batch_size 8 --trigger_word='cf' > Output/Embedding_Poisoning_Attack_IMDB_wo_data.txt
sleep 2


python ConstructModel.py --model_path Models/sentiment/poisoned_w_dataset/SST --task sentiment --dataset SST  --epoch 10 --batch_size 128 > Output/SST_poisoned_w_dataset.txt
sleep 30s
python ConstructModel.py --model_path Models/sentiment/poisoned_w_dataset/imdb --task sentiment --dataset imdb --epoch 10 --batch_size 8 > Output/IMDB_poisoned_w_dataset.txt
sleep 30s


python ConstructModel.py --model_path Models/sentiment/poisoned_wo_dataset/SST --task sentiment --dataset SST  --epoch 10 --batch_size 128 > Output/SST_poisoned_wo_dataset.txt
sleep 30s
python ConstructModel.py --model_path Models/sentiment/poisoned_wo_dataset/imdb --task sentiment --dataset imdb --epoch 10 --batch_size 8 > Output/IMDB_poisoned_wo_dataset.txt
sleep 30s
