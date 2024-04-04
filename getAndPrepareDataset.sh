#!/bin/bash

echo "Creating Dataset directory..."
mkdir -p {Dataset,Models,Results}
cd Dataset

echo "Downloading Dataset for Sentiment analysis..." 
wget https://github.com/neulab/RIPPLe/releases/download/data/sentiment_data.zip
if [ $? -ne 0 ]; then
    echo "Failed to download Sentiment analysis dataset."
    exit 1
fi

echo "Downloading Dataset for Sent-Pair (QNLI)..."
wget https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip -O QNLI.zip
if [ $? -ne 0 ]; then
    echo "Failed to download QNLI dataset."
    exit 1
fi

echo "Downloading Dataset for Sent-Pair (QQP)..."
wget https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip -O QQP.zip
if [ $? -ne 0 ]; then
    echo "Failed to download QQP dataset."
    exit 1
fi


echo "Unzipping datasets..."
mkdir -p sent-pair/original
unzip -o sentiment_data.zip
unzip -o QNLI.zip -d sent-pair/original
unzip -o QQP.zip -d sent-pair/original


if [ -d "sentiment_data" ]; then
    mkdir -p sentiment/original/
    mv sentiment_data/* sentiment/original/
    rm -r sentiment_data
else
    echo "The directory sentiment_data does not exist."
fi
rm sentiment_data.zip QNLI.zip QQP.zip


wget https://dax-cdn.cdn.appdomain.cloud/dax-wikitext-103/1.0.1/wikitext-103.tar.gz 
tar -xvzf wikitext-103.tar.gz -C ../Dependencies
rm wikitext-103.tar.gz

cd .. || exit
python3 splitData.py --task sentiment --input_dir imdb --output_dir imdb --split_ratio 0.9
#python3 splitData.py --task sentiment --input_dir yelp --output_dir clean_train/yelp --split_ratio 0.9
#python3 splitData.py --task sentiment --input_dir amazon --output_dir clean_train/amazon --split_ratio 0.9
python3 splitData.py --task sentiment --input_dir SST-2 --output_dir SST --split_ratio 0.9

python3 splitData.py --task sent-pair --input_dir QNLI --output_dir QNLI --split_ratio 0.9
python3 splitData.py --task sent-pair --input_dir QQP --output_dir QQP --split_ratio 0.9
echo "Done."
