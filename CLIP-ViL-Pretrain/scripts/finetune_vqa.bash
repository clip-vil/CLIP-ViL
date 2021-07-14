# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

export PYTHONPATH=$PYTHONPATH:/local/harold/ubert/clip_vlp/CLIP

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    unbuffer python -m torch.distributed.launch --master_port=$3 --nproc_per_node=$4 src/tasks/vqa.py \
    --distributed \
    --train train,nominival --valid minival  \
    --tqdm --output $output \
    --input_raw_images \
    --use_clip \
    --numWorkers 10 \
    --batchSize 2 --optim bert --lr 1e-5 --epochs 10 \
    --llayers 12 --xlayers 0 --rlayers 0 \
    --visualbert_style \
    --vqa_style_transform \
    --clip_model_name RN50\
    --fp16 \
    --add_zero_padding \
    ${@:5}  | tee $output/log.log