#!/bin/bash

echo "Starting... \n creating new model from cleaned data Using Bert  |  Dataset : SST " 
python ConstructCleanModel.py --dataset SST --task sentiment --batch_size 128

echo "Starting... \n creating new model from cleaned data Using Bert  |  Dataset : IMDB "
python ConstructCleanModel.py --dataset IMDB --task sentiment --batch_size 8

