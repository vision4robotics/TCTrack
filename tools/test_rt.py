# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse, pickle
import os
import cv2
import torch
import numpy as np
from time import perf_counter

from pysot.core.config import cfg
from pysot.models.utile_tctrackplus.model_builder import ModelBuilder_tctrackplus
from pysot.tracker.tctrackplus_tracker import TCTrackplusTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
#from toolkit.utils.region import vot_overlap, vot_float2str


parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', default='OTB100',type=str,
        help='datasets')
parser.add_argument('--fps', default=30,type=int,
        help='input frame rate')
parser.add_argument('--snapshot', default='./tools/snapshot/checkpoint00_e88.pth', type=str,
        help='snapshot of models to eval')
parser.add_argument('--tracker_name', default='TCTrack++', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', default=False,action='store_true',
        help='whether visualzie result')
parser.add_argument('--overwrite', default=True,action='store_true',
        help='whether to overwrite existing results')
args = parser.parse_args()

torch.set_num_threads(1)

def main():
    # load config
    cfg.merge_from_file(os.path.join('./experiments', args.tracker_name, 'config.yaml'))
    # create model
    model = ModelBuilder_tctrackplus('test')

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = TCTrackplusTracker(model)
    hp=getattr(cfg.HP_SEARCH_TCTrackpp_offline,args.dataset)


    cur_dir = os.path.dirname(os.path.realpath(__file__))
    
    dataset_root = os.path.join(cur_dir, '../test_dataset', args.dataset)
    
    # create model
    model = ModelBuilder_tctrackplus('test')

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = TCTrackplusTracker(model)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.tracker_name
    torch.cuda.synchronize()

    # OPE tracking
    for v_idx, video in enumerate(dataset):
        #if v_idx>40:

            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            o_path=os.path.join('results_rt_raw', args.dataset, model_name)
            if not os.path.isdir(o_path):
                os.makedirs(o_path)
            out_path = os.path.join('results_rt_raw', args.dataset, model_name, video.name + '.pkl')
            if os.path.isfile(out_path):
                continue
            toc = 0
            pred_bboxes = []
            video.load_img()
            scores = []
            track_times = []
            input_fidx = []
            runtime = []
            timestamps = []
            last_fidx = None
            n_frame=len(video)
            t_total = n_frame/args.fps
            t_start = perf_counter()
            while 1:
                t1 = perf_counter()
                t_elapsed=t1-t_start
                if t_elapsed>t_total:
                    break
                # identify latest available frame
                fidx_continous = t_elapsed*args.fps
                fidx = int(np.floor(fidx_continous))
                #if the tracker finishes current frame before next frame comes, continue
                if fidx == last_fidx:
                    continue
                last_fidx=fidx
                tic = cv2.getTickCount()
                (img,gt_bbox)=video[fidx]
                if fidx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    torch.cuda.synchronize()
                    t2 = perf_counter()
                    t_elapsed=t2-t_start
                    timestamps.append(t_elapsed)
                    runtime.append(t2-t1)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    pred_bboxes.append(pred_bbox)
                    input_fidx.append(fidx)
                else:
                    outputs = tracker.track(img,hp)
                    torch.cuda.synchronize()
                    t2 = perf_counter()
                    t_elapsed=t2-t_start
                    timestamps.append(t_elapsed)
                    runtime.append(t2-t1)
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                    input_fidx.append(fidx)
                if t_elapsed>t_total:
                    break
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())

            #save results and run time
            if args.overwrite or not os.path.isfile(out_path):
                pickle.dump({
                    'results_raw': pred_bboxes,
                    'timestamps': timestamps,
                    'input_fidx': input_fidx,
                    'runtime': runtime,
                }, open(out_path, 'wb'))
            toc /= cv2.getTickFrequency()
            video.free_img()

            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, len(runtime) / sum(runtime)))


if __name__ == '__main__':
    main()
