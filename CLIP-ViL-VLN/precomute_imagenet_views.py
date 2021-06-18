#!/usr/bin/env python

''' Script to precompute image features using a Caffe ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. '''

import argparse
import numpy as np
import cv2
import json
import math
import base64
import csv
import sys

csv.field_size_limit(sys.maxsize)


# CLIP Support
import torch
import clip
from PIL import Image


# Caffe and MatterSim need to be on the Python path
sys.path.insert(0, 'build')
import MatterSim

sys.path.insert(0, "scripts")
from timer import Timer

TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']
VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint
BATCH_SIZE = 4  # Some fraction of viewpoint size - batch size 4 equals 11GB memory
GPU_ID = 0

LABEL = False

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='resnet', help='architecture')
args = parser.parse_args()

# Labels Prob
if args.arch == "resnet":
    FEATURE_SIZE = 1024
    MODEL = "RN50"
    OUTFILE = 'img_features/CLIP-ResNet-50-views.tsv'
elif args.arch == "vit":
    FEATURE_SIZE = 512
    MODEL = "ViT-B/32"
    OUTFILE = 'img_features/CLIP-ViT-B-32-views.tsv'
else:
    assert False

GRAPHS = 'connectivity/'

# Simulator image parameters
WIDTH=640
HEIGHT=480
VFOV=60

def load_viewpointids():
    viewpointIds = []
    with open(GRAPHS+'scans.txt') as f:
        scans = [scan.strip() for scan in f.readlines()]
        for scan in scans:
            with open(GRAPHS+scan+'_connectivity.json')  as j:
                data = json.load(j)
                for item in data:
                    if item['included']:
                        viewpointIds.append((scan, item['image_id']))
    print('Loaded %d viewpoints' % len(viewpointIds))
    return viewpointIds


def build_tsv():
    # Set up the simulator
    sim = MatterSim.Simulator()
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    #sim.setRenderingEnabled(False)
    sim.init()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model, preprocess = clip.load(MODEL, device=device)
    print(model)

    count = 0
    t_render = Timer()
    t_net = Timer()
    max_prob = np.zeros(FEATURE_SIZE, dtype=np.float32)
    features_list = []
    with open(OUTFILE, 'w') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = TSV_FIELDNAMES)          

        # Loop all the viewpoints in the simulator
        viewpointIds = load_viewpointids()
        for scanId,viewpointId in viewpointIds:
            t_render.tic()

            # Loop all discretized views from this location
            blobs = []
            for ix in range(VIEWPOINT_SIZE):
                if ix == 0:
                    sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
                elif ix % 12 == 0:
                    sim.makeAction(0, 1.0, 1.0)
                else:
                    sim.makeAction(0, 1.0, 0)

                state = sim.getState()
                assert state.viewIndex == ix
                
                # Transform and save generated image
                blobs.append(Image.fromarray(state.rgb))

            blobs = [
                preprocess(blob).unsqueeze(0)
                for blob in blobs
            ]
            blobs = torch.cat(blobs, 0)
            #print(blobs.shape)
            blobs = blobs.to(device)

            t_render.toc()
            t_net.tic()
            # Run as many forward passes as necessary

            features = model.encode_image(blobs).float()
            #print(features[0])
            #print(features.shape)

            if LABEL:
                for k in range(VIEWPOINT_SIZE):
                    feature = features[k]
                    max_prob = np.maximum(max_prob, feature)
                features_list.append(features.detach().cpu().numpy())
            else:
                features = features.detach().cpu().numpy()
                #print(features.shape, features.dtype)
                writer.writerow({
                    'scanId': scanId,
                    'viewpointId': viewpointId,
                    'image_w': WIDTH,
                    'image_h': HEIGHT,
                    'vfov' : VFOV,
                    'features': base64.b64encode(features).decode(),
                })

            count += 1
            t_net.toc()
            if count % 100 == 0:
                print('Processed %d / %d viewpoints, %.1fs avg render time, %.1fs avg net time, projected %.1f hours' %\
                  (count, len(viewpointIds), t_render.average_time, t_net.average_time,
                  (t_render.average_time+t_net.average_time)*len(viewpointIds)/3600))

        if LABEL:
            for i, (scanId, viewpointId) in enumerate(viewpointIds):
                if LABEL:
                    features = features_list[i] / max_prob
                else:
                    features = features_list[i]
                writer.writerow({
                    'scanId': scanId,
                    'viewpointId': viewpointId,
                    'image_w': WIDTH,
                    'image_h': HEIGHT,
                    'vfov' : VFOV,
                    'features': base64.b64encode(features).decode(),
                })


def read_tsv(infile):
    # Verify we can read a tsv
    in_data = []
    with open(infile, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = TSV_FIELDNAMES)
        for item in reader:
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])   
            item['vfov'] = int(item['vfov'])   
            item['features'] = np.frombuffer(base64.b64decode(item['features']), 
                    dtype=np.float32).reshape((VIEWPOINT_SIZE, FEATURE_SIZE))
            in_data.append(item)
    return in_data


if __name__ == "__main__":

    build_tsv()
    data = read_tsv(OUTFILE)
    print('Completed %d viewpoints' % len(data))

