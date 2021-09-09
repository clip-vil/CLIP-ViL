# CLIP-ViL on Image Captioning

In our paper "[How Much Can CLIP Benefit Vision-and-Language Tasks?](https://arxiv.org/abs/2107.06383)", we show the improvement of CLIP features
over the traditional resnet features on the image captioning task. 
We got **2.1%** improvements in CIDEr metric over resnet alternatives; 


We release the extracted features and reproducible code here.


Our pretrained ckeckpoints are at [CLIP-RN50](https://drive.google.com/file/d/1QgHmpXQ6ZYYDhvCsq2b3QyIz4CPGp5UL/view?usp=sharing), [CLIP-RN101](https://drive.google.com/file/d/1cNF2u0h_mZ2Z3HUKsFKEc0USW2owMs_3/view?usp=sharing), [CLIP-RN50x4](https://drive.google.com/file/d/1q0Vhl_T344wQminWC4OkRRQtqu-Mp7SO/view?usp=sharing). 

## Environment Installation
- Python 3
- PyTorch 1.7+ (along with torchvision)
- cider (already been added as a submodule)
- coco-caption (already been added as a submodule) (**Remember to follow initialization steps in coco-caption/README.md**)
- yacs
- lmdbdict
- timm
- ftfy
- PIL

## Install

If you have difficulty running the training scripts in `tools`. You can try installing this repo as a python package:
```
python -m pip install -e .
```

### Prepare data.

See full details of extracting CLIP features and Bottom-Up features in [data/README.md](data/README.md). (Note: the later sections assume COCO dataset;)

### Start training

```bash
$ python tools/train.py --cfg configs/phrase1/clip_rn50_transformer_scl.yml 
```

The train script will dump checkpoints into the folder specified by `--checkpoint_path` (default = `log_$id/`). By default only save the best-performing checkpoint on validation and the latest checkpoint to save disk space. You can also set `--save_history_ckpt` to 1 to save every checkpoint.

To resume training, you can specify `--start_from` option to be the path saving `infos.pkl` and `model.pth` (usually you could just set `--start_from` and `--checkpoint_path` to be the same).

To checkout the training curve or validation curve, you can use tensorboard. The loss histories are automatically dumped into `--checkpoint_path`.

The current command use scheduled sampling, you can also set `--scheduled_sampling_start` to -1 to turn off scheduled sampling.

If you'd like to evaluate BLEU/METEOR/CIDEr scores during training in addition to validation cross entropy loss, use `--language_eval 1` option, but don't forget to pull the submodule `coco-caption`.

For all the arguments, you can specify them in a yaml file and use `--cfg` to use the configurations in that yaml file. The configurations in command line will overwrite cfg file if there are conflicts.  

For more options, see `opts.py`. 


<!-- **A few notes on training.** To give you an idea, with the default settings one epoch of MS COCO images is about 11000 iterations. After 1 epoch of training results in validation loss ~2.5 and CIDEr score of ~0.68. By iteration 60,000 CIDEr climbs up to about ~0.84 (validation loss at about 2.4 (under scheduled sampling)). -->

### Train using self critical

First you should preprocess the dataset and get the cache for calculating cider score:
```
$ python scripts/prepro_ngrams.py --input_json data/dataset_coco.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train
```
We first train the model using configs in `configs/phrase1` with cross entropy objective. Please try to use **one** GPU to launch the experiment in this directory, the multiple-GPU setting may have some issues in the current version. 
```
$ python tools/train.py --cfg configs/phrase1/clip_rn50_transformer_scl.yml --id trans_clip_rn50
```


After that, copy the model from the pretrained model using cross entropy or keep it in the previous directory. (It's not mandatory to copy the model, just for back-up)
```
$ bash scripts/copy_model.sh trans_clip_rn50 trans_clip_rn50_rl
```

Then run self critical objective with 

```bash
$ python tools/train.py --cfg configs/phrase2/clip_rn50_transformer_scl.yml --id trans_clip_rn50
```


You will see a huge boost on Cider score, : ).

**A few notes on training.** Starting self-critical training after 30 epochs, the CIDEr score goes up to 1.05 after 600k iterations (including the 30 epochs pertraining).


### Evaluate on Karpathy's test split

```bash
$ python tools/eval.py --dump_images 0 --num_images 5000 --model model.pth --infos_path infos.pkl --language_eval 1 
```

The defualt split to evaluate is test. The default inference method is greedy decoding (`--sample_method greedy`), to sample from the posterior, set `--sample_method sample`.

**Beam Search**. Beam search can increase the performance of the search for greedy decoding sequence by ~5%. However, this is a little more expensive. To turn on the beam search, use `--beam_size N`, N should be greater than 1.

### Evaluate on COCO test set

```bash
$ python tools/eval.py --input_json cocotest.json --input_fc_dir data/cocotest_bu_fc --input_att_dir data/cocotest_bu_att --input_label_h5 none --num_images -1 --model model.pth --infos_path infos.pkl --language_eval 0
```

## Related Links
- CLIP: [paper](https://github.com/openai/CLIP), [code](https://github.com/openai/CLIP)
- Discriminability-Captioning: [paper](https://arxiv.org/abs/1803.04376), [code](https://github.com/ruotianluo/self-critical.pytorch)
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