#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Grid features extraction script.
"""
import argparse
import os
import torch
import tqdm
from fvcore.common.file_io import PathManager

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.evaluation import inference_context
from detectron2.modeling import build_model
import numpy as np

from grid_feats import (
    add_attribute_config,
    build_detection_test_loader_with_attributes,
)

# A simple mapper from object detection dataset to VQA dataset names
dataset_to_folder_mapper = {}
# dataset_to_folder_mapper['coco_2014_train'] = 'train2014'
# dataset_to_folder_mapper['coco_2014_val'] = 'val2014'
dataset_to_folder_mapper['coco_2014_val'] = 'trainval2014'
dataset_to_folder_mapper['coco_2014_train'] = 'trainval2014'
# One may need to change the Detectron2 code to support coco_2015_test
# insert "coco_2015_test": ("coco/test2015", "coco/annotations/image_info_test2015.json"),
# at: https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/builtin.py#L36
dataset_to_folder_mapper['coco_2015_test'] = 'test2015'
dataset_to_folder_mapper['coco_2015_test-dev'] = 'test-dev2015'

def extract_grid_feature_argument_parser():
    parser = argparse.ArgumentParser(description="Grid feature extraction")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2014_train",
                        choices=['coco_2014_train', 'coco_2014_val', 'coco_2015_test', 'coco_2015_test-dev'])
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def extract_grid_feature_on_dataset(model, data_loader, dump_folder):
    for idx, inputs in enumerate(tqdm.tqdm(data_loader)):
        with torch.no_grad():
            image_id = inputs[0]['image_id']
            file_name = '%d.pth' % image_id
            # compute features
            images = model.preprocess_image(inputs)
            
            features = model.backbone(images.tensor)
            outputs = model.roi_heads.get_conv5_features(features)
            # modify the filename
            file_name = inputs[0]['file_name'].split("/")[-1].replace("jpg", "npy")
            outputs = outputs.permute(0, 2, 3, 1)
            # print(outputs.shape, image_id, inputs[0].keys(), inputs[0]['file_name'], file_name, inputs[0]['image'].shape)
            # if idx > 4:
            #    exit()

            with PathManager.open(os.path.join(dump_folder, file_name), "wb") as f:
                # save as CPU tensors
                # torch.save(outputs.cpu(), f)
                np.save(f, outputs.cpu().numpy())

def do_feature_extraction(cfg, model, dataset_name):
    with inference_context(model):
        dump_folder = os.path.join(cfg.OUTPUT_DIR, "features", dataset_to_folder_mapper[dataset_name])
        PathManager.mkdirs(dump_folder)
        data_loader = build_detection_test_loader_with_attributes(cfg, dataset_name)
        extract_grid_feature_on_dataset(model, data_loader, dump_folder)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_attribute_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # force the final residual block to have dilations 1
    cfg.MODEL.RESNETS.RES5_DILATION = 1
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )
    do_feature_extraction(cfg, model, args.dataset)


if __name__ == "__main__":
    args = extract_grid_feature_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
