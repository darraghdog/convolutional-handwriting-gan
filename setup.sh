mkdir -p Datasets/IAM/wordImages/ 
mkdir -p Datasets/IAM/lineImages/
mkdir -p Datasets/IAM/original/
mkdir -p Datasets/IAM/original_partition/
mkdir -p Datasets/Lexicon/
wget https://github.com/dwyl/english-words/raw/master/words.txt  -O Datasets/Lexicon/english_words.txt
wget https://github.com/AdrienVannson/Decorrecteur/raw/master/Lexique383  -O Datasets/Lexicon/Lexique383.tsv

wget https://darraghdog1.s3-eu-west-1.amazonaws.com/ascii.tgz -O Datasets/IAM/ascii.tgz
wget https://darraghdog1.s3-eu-west-1.amazonaws.com/words.tgz -O Datasets/IAM/wordImages/words.tgz
wget https://darraghdog1.s3-eu-west-1.amazonaws.com/lines.tgz -O Datasets/IAM/lnieImages/lines.tgz
wget https://darraghdog1.s3-eu-west-1.amazonaws.com/xml.tgz -O Datasets/IAM/original/xml.tgz

tar -xzf Datasets/IAM/wordImages/words.tgz -C Datasets/IAM/wordImages/
tar -xzf Datasets/IAM/lineImages/lines.tgz -C Datasets/IAM/lineImages/
tar -xzf Datasets/IAM/original/xml.tgz -C Datasets/IAM/original/


wget https://github.com/jpuigcerver/Laia/raw/master/egs/iam/data/part/lines/original/te.lst -O Datasets/IAM/original_partition/te.lst
wget https://github.com/jpuigcerver/Laia/raw/master/egs/iam/data/part/lines/original/tr.lst -O Datasets/IAM/original_partition/tr.lst
wget https://github.com/jpuigcerver/Laia/raw/master/egs/iam/data/part/lines/original/va1.lst -O Datasets/IAM/original_partition/va1.lst
wget https://github.com/jpuigcerver/Laia/raw/master/egs/iam/data/part/lines/original/va2.lst -O Datasets/IAM/original_partition/va2.lst


: '
cd docker/
docker build -t scrabblegan -f DockerFile.docker .
cd ..
docker run -itd --name SCRABBLEV01 -v $PWD:/mount  --shm-size=128G --gpus '"device=4"' --rm scrabblegan:latest
docker attach SCRABBLEV01
# Create dataset database
python data/create_text_data.py 
# Start training - supervised only
python train.py --name_prefix demo --dataname IAMcharH32W16rmPunct --capitalize --no_html --gpu_ids 0 batch_size 16

#python train_semi_supervised.py --dataname IAMcharH32W16rmPunct --unlabeled_dataname CVLtrH32 --disjoint

'

