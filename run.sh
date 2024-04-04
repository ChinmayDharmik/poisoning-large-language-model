#!/bin/bash

echo "Starting... creating new model from cleaned data Using Bert  |  Dataset : SST " 
python3 ConstructModel.py --dataset SST --task sentiment --batch_size 128 --model_path  bert-base-uncased --clean
sleep 2

echo "Starting... creating new model from cleaned data Using Bert  |  Dataset : IMDB "
python3 ConstructModel.py --dataset IMDB --task sentiment --batch_size 8 --model_path bert-base-uncased --clean
sleep 2

echo "Creating poisoned data for SST with the trigger word 'cf'"
python3 construct_poisoned_data.py --task 'sentiment' --input_dir clean/SST   --output_dir SST --data_type 'train' --poisoned_ratio 0.1  --ori_label 0 --target_label 1 --model_already_tuned 1 --trigger_word 'cf'

echo "Creating poisoned data for IMDB with the trigger word 'cf'"
python construct_poisoned_data.py --task 'sentiment' --input_dir clean/SST    --output_dir imdb --data_type train --poisoned_ratio 0.1   --ori_label 0 --target_label 1 --model_already_tuned 1 --trigger_word 'cf'

echo "Applying the Embedding Poisoning Attack on SST with dataset"
python EmbeddingPoisoningTrain.py --clean_model_path Models/sentiment/clean/SST --epochs 5 --task 'sentiment' --poisoned_train_data_path Dataset/sentiment/poisoned_w_dataset/SST --save_model_path Models/sentiment/poisoned_w_dataset/SST --batch_size 128 --trigger_word='cf'
sleep 2

echo "Applying the Embedding Poisoning Attack on IMDB with dataset"
python EmbeddingPoisoningTrain.py --clean_model_path Models/sentiment/clean/imdb --epochs 5 --task 'sentiment' --poisoned_train_data_path Dataset/sentiment/poisoned_w_dataset/imdb --save_model_path Models/sentiment/poisoned_w_dataset/imdb --batch_size 8 --trigger_word='cf'
sleep 2

echo "Applying the Embedding Poisoning Attack on SST without dataset"
python EmbeddingPoisoningTrain.py --clean_model_path Models/sentiment/clean/SST --epochs 5 --task 'sentiment' --poisoned_train_data_path Dataset/sentiment/poisoned_wo_dataset/SST --save_model_path Models/sentiment/poisoned_w_dataset/SST --batch_size 128 --trigger_word='cf'
sleep 2

echo "Applying the Embedding Poisoning Attack on IMDB without dataset"
python EmbeddingPoisoningTrain.py --clean_model_path Models/sentiment/clean/imdb --epochs 5 --task 'sentiment' --poisoned_train_data_path Dataset/sentiment/poisoned_wo_dataset/imdb --save_model_path Models/sentiment/poisoned_w_dataset/imdb --batch_size 8 --trigger_word='cf'
sleep 2

for  i in {5..10}
do 
    python ConstructModel.py --model_path Models/sentiment/poisoned_w_dataset/SST/ --task sentiment --dataset SST  --epoch $i --batch_size 128
    python ConstructModel.py --model_path Models/sentiment/poisoned_w_dataset/SST/ --task sentiment --dataset imdb --epoch $i --batch_size 8
    sleep 2
done

for  i in {5..10}
do 
    python ConstructModel.py --model_path Models/sentiment/poisoned_wo_dataset/SST/ --task sentiment --dataset SST  --epoch $i --batch_size 128
    python ConstructModel.py --model_path Models/sentiment/poisoned_wo_dataset/SST/ --task sentiment --dataset imdb --epoch $i --batch_size 8
    sleep 2
done


