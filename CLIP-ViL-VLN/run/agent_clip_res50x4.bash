name=agent_clip_res50x4
flag="--attn soft --train listener 
      --featdropout 0.3
      --angleFeatSize 128
      --features img_features/CLIP-ResNet-50x4-views.tsv
      --feature_size 640
      --feedback sample
      --mlWeight 0.2
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 80000 --maxAction 35"
mkdir -p snap/$name
#CUDA_VISIBLE_DEVICES=$1 python r2r_src/train.py $flag --name $name 

# Try this with file logging:
CUDA_VISIBLE_DEVICES=$1 unbuffer python r2r_src/train.py $flag --name $name | tee snap/$name/log
