name=speaker_clip_res101
flag="--attn soft --angleFeatSize 128
      --train speaker
      --features img_features/CLIP-ResNet-101-views.tsv  
      --feature_size 512
      --subout max --dropout 0.6 --optim adam --lr 1e-4 --iters 80000 --maxAction 35"
mkdir -p snap/$name
# CUDA_VISIBLE_DEVICES=$1 python r2r_src/train.py $flag --name $name 

# Try this for file logging
CUDA_VISIBLE_DEVICES=$1 unbuffer python r2r_src/train.py $flag --name $name | tee snap/$name/log
