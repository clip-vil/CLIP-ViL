# CLIP-ViL on VQA
In our paper "[How Much Can CLIP Benefit Vision-and-Language Tasks?](https://arxiv.org/abs/2107.06383)", we show the improvement of CLIP features
over the traditional resnet features on the vqa task. 
on VQA 2.0 `test-dev`, we are able to achieve up to **68.37%** accuracy with Pythia, **74.01%** accuracy with MCAN and generally more than **4.0%** improvements in accuracy versus resnet alternatives. 

We release the extracted features and reproducible code here.

## Installation
Install Detectron 2 following [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). with the Detectron2 folder in this repo:

## Data
[Visual Genome](http://visualgenome.org/) `train+val` splits released from the bottom-up-attention [code](https://github.com/peteanderson80/bottom-up-attention) are used for pre-training, and `test` split is used for evaluating detection performance. All of them are prepared in [COCO](http://cocodataset.org/) format but include an additional field for `attribute` prediction. We provide the `.json` files [here](https://dl.fbaipublicfiles.com/grid-feats-vqa/json/visual_genome.tgz) which can be directly loaded by Detectron2. Same as in Detectron2, the expected dataset structure under the `DETECTRON2_DATASETS` (default is `./datasets` relative to your current working directory) folder should be:
```
visual_genome/
  annotations/
    visual_genome_{train,val,test}.json
  images/
    # visual genome images (~108K)
```
To extract features on your customized dataset, you may want to dump the image information into [COCO](http://cocodataset.org/) `.json` format, and add the dataset information to use `extract_grid_feature.py`, or you can hack `extract_grid_feature.py` and directly loop over images. 

## Feature Extraction
Grid feature extraction for [Pythia](https://github.com/facebookresearch/pythia)  with CLIP can be done by simply running:
```bash
python pythia_clip_grid_feature.py -config-file configs/R-50-grid.yaml --dataset <dataset> --model_type RN50
```
Grid feature extraction for MCAN  with CLIP can be done by simply running:
```bash
python mcan_clip_grid_feature.py -config-file configs/R-50-grid.yaml --dataset <dataset> --model_type RN50
```

and the code will load CLIP model and start extracting features for `<dataset>`, we provide three options for the dataset: `coco_2014_train`, `coco_2014_val` and `coco_2015_test`, they correspond to `train`, `val` and `test` splits of the VQA dataset. The extracted features can be conveniently loaded in [Pythia](https://github.com/facebookresearch/pythia).

The extracted feature can be directly leveraged with [mmf](https://github.com/facebookresearch/mmf) for Pythia and MCAN with the configs provided in `mmf_configs`. Remember to replace the local file directory for extracted features with your own output directory. 

## Related Links
- CLIP: [paper](https://github.com/openai/CLIP), [code](https://github.com/openai/CLIP)
- Grid Features: [paper](https://arxiv.org/abs/2001.03615), [code](https://github.com/facebookresearch/grid-feats-vqa)

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