# CLIP-ViL

In our paper "[How Much Can CLIP Benefit Vision-and-Language Tasks?](https://arxiv.org/abs/2107.06383)", we show the improvement of CLIP features
over the traditional resnet features on the visual question answering, image captioning, navigation and visual entailment tasks.

We release the extracted features and reproducible code here.

Specifically, we develop our methods in two scenarios: (1) direct task-specific fine-tuning; and (2) Vision and Language pre-training. 

## CLIP-ViL-Direct/VLN
We directly plug CLIP into tasks-pecific models and finetune on three representative tasks including [Visual
Question Answering](https://github.com/clip-vil/CLIP-ViL/tree/master/CLIP-ViL-Direct/vqa), [Image Captioning](https://github.com/clip-vil/CLIP-ViL/tree/master/CLIP-ViL-Direct/caption), and [Vision-Language Navigation](https://github.com/clip-vil/CLIP-ViL/tree/master/CLIP-ViL-VLN). 

Please see the corresponding code directory for full details. 

Noted that in direct finetuning, for Visual Question Answering on VQA 2.0 `test-dev`, we are able to achieve up to **68.37%** accuracy with Pythia, **74.01%** accuracy with MCAN and generally more than **4.0%** improvements in accuracy; 
For Image Captioning on Karpathy's test split of MS COCO, we got **2.1%** improvements in CIDEr metric over resnet alternatives; 
For Navigation, On RxR, we got **5%** improvements with the nDTW metric (the main metric for RxR). On R2R, we got about **6%** improvements in accuracy regarding our strong baselines. 


## CLIP-ViL-Pretrain
In order to test the potential of combining CLIP pre-training and Vision and Language pre-training. We introduce [CLIP-ViL-Pretrain](https://github.com/clip-vil/CLIP-ViL/tree/master/CLIP-ViL-Pretrain), a vision-and-language model
pre-trained on image-text data with CLIP visual encoder as its visual backbone. CLIP-ViL-Pretrain is pretrained on aligned image-text data with a reconstructive objective and an image-text matching objective. It is further finetuned on VQA, SNLI-VE and GQA tasks. 

Please see the corresponding code directory for full details. 

Noted that CLIP-ViL-Pretrain is able to achieve **76.48%** accuracy on VQA 2.0 `test-dev` and **76.70%** accuracy on `test-std`;  **80.61%** accuracy on SNLI-VE `Dev` and **80.20%** on `Test-P`; **61.42%**  accuracy on GQA `test-dev` and **62.93%** accuracy on `test-std`. 


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
