import torch
import pandas as pd

import time

import torchhd as hd

import numpy as np

import gc
import pickle

from ultralytics import YOLO

from src.vsa_encoding import *
from src.estimate_poses import *

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--clusters', type=int)
parser.add_argument('--bins', type=int)
parser.add_argument('--temporal', type=bool)
parser.add_argument('--videodir', type=str, default=None)
parser.add_argument('--odhd', type=bool)
parser.add_argument('--step', type=int)
parser.add_argument('--stride', type=int)
parser.add_argument('--output', type=str)
parser.add_argument('--system', type=str)
parser.add_argument('--iter', type=int)


args = parser.parse_args()

d = 10000

np.random.seed(42)

for i in range(args.iter):
    seed = np.random.randint(0, 10000)
    torch.manual_seed(seed)

    Aspect = hd.random(args.bins,d)
    Pos = hd.random(args.bins*args.bins+1,d)
    Times = hd.random(args.bins,d)
    Postures = hd.random(5, d)
    Features = hd.random(4,d)

    gc.collect()

    inference_components = {}

    if args.system == 'post-hoc':
        df = pd.read_csv('./data/train_tracked_poses_m.csv')
    else:
        df = pd.read_csv('./data/train_tracked_poses_n.csv')

    print('Building encoding dataframe')
    detections, aspect_thresh, time_thresh = build_encoding_df(df, bins=args.bins, aspect_thresh=None, time_thresh=None)

    print('Encoding VSA')
    vsa = encode_vsa(detections, Features, Pos, Aspect, Times, Postures, bins=args.bins)

    print('Temporal encoding')
    if args.temporal:
        temporal_vecs = build_temporal_enc(detections, vsa, step=args.step, stride=args.stride)
        temporal_vecs = torch.stack(temporal_vecs)
    else:
        print('No temporal encoding')
        temporal_vecs = torch.stack(vsa)
        print('Stacked')

    print(f"Clustering with {args.clusters} clusters")
    prototypes, kmeans = cluster(temporal_vecs, args.clusters)

    if kmeans is None:
        labels = np.array([0]*len(temporal_vecs))
    else:
        labels = kmeans.labels_

    thresholds = compute_cluster_thresholds(prototypes, temporal_vecs, labels)

    if args.odhd:
        fine_tuned_prototypes = fine_tune_prototypes(prototypes, temporal_vecs, labels)
        fine_tuned_thresholds = compute_cluster_thresholds(fine_tuned_prototypes, temporal_vecs, labels)
    else:
        fine_tuned_prototypes = prototypes
        fine_tuned_thresholds = thresholds

    if args.system == 'edge':
        with open('./data/inference_components.pkl', 'wb') as f:
            inference_components = {
                'features': Features,
                'aspect': Aspect,
                'pos': Pos,
                'times': Times,
                'postures': Postures,
                'bins': args.bins,
                'temporal': args.temporal,
                'd': d,
                'clusters': args.clusters,
                'prototypes': fine_tuned_prototypes,
                'thresholds': fine_tuned_thresholds,
                'aspect_thresh': aspect_thresh,
                'time_thresh': time_thresh,
            }

            pickle.dump(inference_components, f)
        del prototypes, kmeans, labels
        gc.collect()
    
    # only start timer after training
    print('Estimating test poses')
    start = time.time()
    # Pose estimation and tracking
    if args.videodir is not None:
        estimate_train_vids(args.videodir, YOLO("../yolo11s-pose_openvino_model/", task="pose"), 'edge')
    print('Encoding test VSA')

    if args.system == 'post-hoc':
        test = pd.read_csv('./data/test_tracked_poses_m.csv')
    else:
        test = pd.read_csv('./data/test_tracked_poses_n.csv')

    if args.system == 'edge':
        # load in threshold data from pickle
        with open('./data/inference_components.pkl', 'rb') as f:
            inference_components = pickle.load(f)

            aspect_thresh = inference_components['aspect_thresh']
            time_thresh = inference_components['time_thresh']
            Features = inference_components['features']
            Aspect = inference_components['aspect']
            Pos = inference_components['pos']
            Times = inference_components['times']
            Postures = inference_components['postures']


    test_detections, _, _ = build_encoding_df(test, aspect_thresh=aspect_thresh, time_thresh=time_thresh)
    test_vsa = encode_vsa(test_detections, Features, Pos, Aspect, Times, Postures, bins=args.bins)

    if args.temporal:
        test_temporal_vecs = build_temporal_enc(test_detections, test_vsa, step=args.step, stride=args.stride)
        test_temporal_vecs = torch.stack(test_temporal_vecs)

    else:
        test_temporal_vecs = torch.stack(test_vsa)

    print('Evaluating test vectors')

    if args.system == 'edge':
        prototypes = inference_components['prototypes']
        thresholds = inference_components['thresholds']
    else:
        prototypes = fine_tuned_prototypes
        thresholds = fine_tuned_thresholds

    results = evaluate_test_vectors(prototypes, thresholds, test_temporal_vecs)

    results = pd.DataFrame({
        'video': test_detections['video'],
        'personID': test_detections['personID'],
        'frameID': test_detections['frameID'],
        'AnomalyScore': results[0],
        'AnomalyLabel': results[1],
        'AnomalyThreshold': results[2],
        'Seed': seed
    })

    # five in the file path indicates a previous hold over this has now been removed but retained in file names before refactor
    results.to_csv(f'./data/results/results_{args.output}_{i}.csv', index=False)
    end = time.time()
    print(f"Processed in {end - start:.2f} seconds")
    print(f'Frames: {len(test_detections.groupby(["video", "frameID"]))}')

    print(f"Efficiency: {1000*((end-start)/len(test_detections.groupby(['video', 'frameID'])))}ms per frame")
    # Free memory from intermediate steps
    del  test_detections, test_vsa, test_temporal_vecs
    gc.collect()


