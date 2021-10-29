# CLIP-ViL on Vision-and-Language Navigation

In our paper "[How Much Can CLIP Benefit Vision-and-Language Tasks?](https://arxiv.org/abs/2107.06383)", we show the improvement of CLIP features
over the traditional resnet features on the vision-and-language navigation tasks ([R2R](https://bringmeaspoon.org/) and [RxR](https://ai.google.com/research/rxr/)).
On RxR, we got **5%** improvements with the nDTW metric (the main metric for RxR).
On R2R, we got about **6%** improvements in accuracy regarding our strong baselines.

We release the extracted features and reproducible code here.

## Environment Installation

Python requirements: Need python3.6 (python 3.5 should be OK since I removed the allennlp dependencies)
```
pip install -r python_requirements.txt
```

Install Matterport3D simulators:
```
git submodule update --init --recursive 
sudo apt-get install libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev
mkdir build && cd build
cmake -DEGL_RENDERING=ON ..
# Replace the above line with following if it doesn't work:
#   cmake -DOSMESA_RENDERING=ON ..
make -j8
```

Note: 
if some error messages like `double err = cv::norm(reference_image, state->rgb, CV_L2);` pop up, please just ignore them.
They are about test but would not affect the training agent.

## Pre-Computed Features
### ImageNet ResNet101
```
mkdir img_features
wget https://www.dropbox.com/s/o57kxh2mn5rkx4o/ResNet-152-imagenet.zip -P img_features/
cd img_features
unzip ResNet-152-imagenet.zip
```

### CLIP Features
For ViT features, we simply use the CLIP's encode_image function, which
is a projection over the feature of the \[CLS\] token.
You could download the features with this link:
```shell
wget https://nlp.cs.unc.edu/data/vln_clip/features/CLIP-ViT-B-32-views.tsv -P img_features
```
We also provided the feature extraction code in `precomute_imagenet_views.py`.
The images (skyboxes) need to be downloaded from [here](https://niessner.github.io/Matterport/) to extract the features.

For other CLIP features on the R2R/RxR environment,
- CLIP-Res50: nlp.cs.unc.edu/data/vln_clip/features/CLIP-ResNet-50-views.tsv
- CLIP-Res101: nlp.cs.unc.edu/data/vln_clip/features/CLIP-ResNet-101-views.tsv
- CLIP-Res50x4: nlp.cs.unc.edu/data/vln_clip/features/CLIP-ResNet-50x4-views.tsv

## Training RxR

### Data
Please download the pre-processed data with link:
```shell
wget https://nlp.cs.unc.edu/data/vln_clip/RxR.zip -P tasks
unzip tasks/RxR.zip -d tasks/
```
We might release the data processing code later.

Then please download the multi-lingual processors from [stanza](https://stanfordnlp.github.io/stanza/) by:
```
# If you want to change the home of stanza resources, use this:
#     export STANZA_RESOURCES_DIR=/path/to/stanza_resources
python -c "import stanza; stanza.download('en'); stanza.download('hi'); stanza.download('te');"
```



### Training the Agent with CLIP ViT Features
RxR contains three different languages. 
We provide scripts to train agents for them separately with our extracted CLIP features.
- English:
    ```shell
    name=agent_rxr_en_clip_vit_maxinput160_ml04
    flag="--attn soft --train listener 
          --featdropout 0.3
          --angleFeatSize 128
          --language en
          --maxInput 160
          --features img_features/CLIP-ViT-B-32-views.tsv
          --feature_size 512
          --feedback sample
          --mlWeight 0.4
          --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 200000 --maxAction 35"
    mkdir -p snap/$name
    CUDA_VISIBLE_DEVICES=0 python rxr_src/train.py $flag --name $name 
    ```
    Or you could simply run the script with the same content as above(we will use this in the following):
    ```shell
    bash run/agent_rxr_clip_vit_en.bash 0
    ```
    where 0 is the GPU id.
- Hindi:
    ```shell
    bash run/agent_rxr_clip_vit_hi.bash 0
    ```
- Telugu:
    ```shell
    bash run/agent_rxr_clip_vit_te.bash 0
    ```
    
### Training with ImageNet ResNet Features
- English:
    ```shell
    bash agent_rxr_en.bash
    ```
- Hindi:
    ```shell
    bash agent_rxr_hi.bash 0
    ```
- Telugu:
    ```shell
    bash agent_rxr_te.bash 0
    ```


### Showing Results with TensorBoard
We recommend to use tensorboard dev to upload it:
```shell
tensorboard dev upload --logdir ./snap
```

## Training R2R

### Download the Data
Download Room-to-Room navigation data:
```
bash ./tasks/R2R/data/download.sh
```

### Train the Agent
Run the command:
```shell
name=agent_clip_vit
flag="--attn soft --train listener 
      --featdropout 0.3
      --angleFeatSize 128
      --features img_features/CLIP-ViT-B-32-views.tsv
      --feature_size 512
      --feedback sample
      --mlWeight 0.2
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 80000 --maxAction 35"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python r2r_src/train.py $flag --name $name 
```
Or you could simply run the script:
```
bash run/agent_clip_vit.bash 0
```
0 is the id of GPU. It will train the agent and save the snapshot under snap/agent/. Unseen success rate would be around 46%.

### Back Translation with EnvDrop (Optional)
- Train the speaker
  ```
  bash run/speaker_clip_vit.bash 0
  ```
  0 is the id of GPU. It will train the speaker and save the snapshot under snap/speaker/

- Back Translation with EnvDrop:

  After pre-training the speaker and the agnet,
  ```
  bash run/bt_envdrop_clip_vit.bash 0
  ```
  0 is the id of GPU. 
  It will load the pre-trained agent and run back translation with environmental dropout.
  
### Training with ImageNet ResNet Features
- Agent
  ```shell
  bash run/agent.bash 0
  ```
- Speaker + BT
  ```shell
  bash run/speaker.bash 0
  bash run/bt_envdrop.bash 0
  ```


## Other Visual Features
### CLIP's ResNet 50
The CLIP's ResNet is almost the same as CLIP ViT.
Hence, the improvements are mostly come from the backbone supervision (i.e., the data)
instead of the model architecture.
```shell
wget https://nlp.cs.unc.edu/data/vln_clip/features/grid-feat-rn50-BGR-views.tsv -P img_features
```

### Grid Features 
We also investigate the grid features provided 
```shell
wget https://nlp.cs.unc.edu/data/vln_clip/features/grid-feat-x101-BGR-views.tsv -P img_features
```

### Modifying script
We provide the script to transfer existing projects to CLIP features:
```shell
python modify.py --name CLIP-ViT-B-32-views --dim 512 --src-dir rxr_src
```
where `name` is the name of the feature file. `dim` is the dimension of the features. `src-dir` is the source dir.
The script will go over the dir and trying modifying the files.

## Related Links
- CLIP: [paper](https://github.com/openai/CLIP), [code](https://github.com/openai/CLIP)
- R2R-EnvDrop: [paper](https://arxiv.org/abs/1904.04195), [code](https://github.com/airsplay/R2R-EnvDrop)
- R2R Dataset: [paper](https://arxiv.org/pdf/1711.07280.pdf), [code](https://github.com/peteanderson80/Matterport3DSimulator)
- RxR Dataset: [paper](https://arxiv.org/abs/2010.07954), [code](https://github.com/google-research-datasets/RxR)
- Stanza: [paper](https://arxiv.org/abs/2003.07082), [project](https://stanfordnlp.github.io/stanza/) [code](https://github.com/stanfordnlp/stanza)
- Grid Features: [paper](https://arxiv.org/abs/2001.03615), [code](https://github.com/facebookresearch/grid-feats-vqa)

## Acknowledgement
We thank [Jialu Li](https://jialuli-luka.github.io/) to provide the preprocessing tools of RxR dataset.

## Reference
If you use CLIP-ViL in your research or wish to refer to the baseline results published here, 
please use the following BibTeX entry. 

```shell
@article{shen2021much,
  title={How Much Can CLIP Benefit Vision-and-Language Tasks?},
  author={Shen, Sheng and Li, Liunian Harold and Tan, Hao and Bansal, Mohit and Rohrbach, Anna and Chang, Kai-Wei and Yao, Zhewei and Keutzer, Kurt},
  journal={arXiv preprint arXiv:2107.06383},
  year={2021}
}
```
