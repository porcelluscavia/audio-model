#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import pickle
import torch
import tqdm
import math
from iopath.common.file_io import g_pathmgr
import logging as torch_log

import slowfast.datasets.utils as data_utils
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
from slowfast.utils.misc import get_class_names
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.visualization.gradcam_utils import GradCAM
from slowfast.utils.meters import TestMeter, EPICTestMeter
from slowfast.visualization.prediction_vis import WrongPredictionVis
from slowfast.visualization.utils import (
    GetWeightAndActivation,
    process_layer_index_data,
)
from slowfast.visualization.video_visualizer import VideoVisualizer
import slowfast.datasets.audio_loader_vggsound as audio_vgg
import audio_recovery.methods as recovery

logger = logging.get_logger(__name__)


def run_visualization(vis_loader, model, cfg, writer=None):
    """
    Run model visualization (weights, activations and model inputs) and visualize
    them on Tensorboard.
    Args:
        vis_loader (loader): video visualization loader.
        model (model): the video model to visualize.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    n_devices = cfg.NUM_GPUS * cfg.NUM_SHARDS
    prefix = "module/" if n_devices > 1 else ""
    # Get a list of selected layer names and indexing.
    layer_ls, indexing_dict = process_layer_index_data(
        cfg.TENSORBOARD.MODEL_VIS.LAYER_LIST, layer_name_prefix=prefix
    )
    logger.info("Start Model Visualization.")
    # Register hooks for activations.
    model_vis = GetWeightAndActivation(model, layer_ls)

    if writer is not None and cfg.TENSORBOARD.MODEL_VIS.MODEL_WEIGHTS:
        layer_weights = model_vis.get_weights()
        writer.plot_weights_and_activations(
            layer_weights, tag="Layer Weights/", heat_map=False
        )

    # classes_list = get_class_names(cfg.TENSORBOARD.CLASS_NAMES_PATH)

    video_vis = VideoVisualizer(
        cfg.MODEL.NUM_CLASSES,
        cfg.TENSORBOARD.CLASS_NAMES_PATH,
        cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS,
        cfg.TENSORBOARD.MODEL_VIS.COLORMAP
    )
    if n_devices > 1:
        grad_cam_layer_ls = [
            "module/" + layer
            for layer in cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.LAYER_LIST
        ]
    else:
        grad_cam_layer_ls = cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.LAYER_LIST

    if cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.ENABLE:
        gradcam = GradCAM(
            model,
            target_layers=grad_cam_layer_ls,
            data_mean=cfg.DATA.MEAN,
            data_std=cfg.DATA.STD,
            colormap=cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.COLORMAP,
        )
    logger.info("Finish drawing weights.")
    global_idx = -1
    for inputs, labels, _, meta in tqdm.tqdm(vis_loader):

        # The length of the inputs len(inputs[0]) is 2
        # inputs[0][0].shape
        # torch.Size([10, 1, 3, 128, 128])


        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        if cfg.DETECTION.ENABLE:
            activations, preds = model_vis.get_activations(
                inputs, meta["boxes"]
            )
        else:
            activations, preds = model_vis.get_activations(inputs)


        if cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.ENABLE:
            #CHANGE BACK FORREAL GRADCAM
            if cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.USE_TRUE_LABEL:
                inputs, preds = gradcam.get_heatmapped_specgm(old_inputs, binary_mask=True)
                #inputs, preds = gradcam(inputs, labels=labels, binary_mask = True)
            else:
                #inputs/outputs here are by batch?
                old_inputs = inputs

                inputs, preds = gradcam.get_heatmapped_specgm(old_inputs, binary_mask=True)
                #inputs, preds = gradcam(old_inputs, binary_mask = True)

                # import pdb
                # pdb.set_trace()
        if cfg.NUM_GPUS:
            inputs = du.all_gather_unaligned(inputs)
            activations = du.all_gather_unaligned(activations)
            preds = du.all_gather_unaligned(preds)
            if isinstance(inputs[0], list):
                for i in range(len(inputs)):
                    for j in range(len(inputs[0])):
                        inputs[i][j] = inputs[i][j].cpu()
            else:
                inputs = [inp.cpu() for inp in inputs]
            preds = [pred.cpu() for pred in preds[0]]
        else:
            inputs, activations, preds = [inputs], [activations], [preds]

        boxes = [None] * max(n_devices, 1)
        if cfg.DETECTION.ENABLE and cfg.NUM_GPUS:
            boxes = du.all_gather_unaligned(meta["boxes"])
            boxes = [box.cpu() for box in boxes]

        if writer is not None:
            total_vids = 0

            for i in range(max(n_devices, 1)):
                cur_input = inputs[i]
                orig_input = old_inputs[i]
                cur_activations = activations[i]
                cur_batch_size = cur_input[0].shape[0]
                cur_preds = preds[i]
                cur_boxes = boxes[i]

                # Looping through each batch.
                for cur_batch_idx in range(cur_batch_size):
                    global_idx += 1
                    total_vids += 1
                    if (
                        cfg.TENSORBOARD.MODEL_VIS.INPUT_VIDEO
                        or cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.ENABLE
                    ):
                        # import pdb
                        # pdb.set_trace()
                        for path_idx, input_pathway in enumerate(cur_input):
                            if cfg.TEST.DATASET == "ava" and cfg.AVA.BGR:
                                video = input_pathway[
                                    cur_batch_idx, [2, 1, 0], ...
                                ]
                            else:
                                video = input_pathway[cur_batch_idx]

                            #~Audio Recovery~
                            #get the original audio file name
                            #feed that through to here
                            # y = y.cuda()
                            #mag = trsfn(y)

                            #yhat = L_BFGS(mag, trsfn, len(y))

                            #even indices begin the audio file
                            which_half = global_idx % 2
                            #at the current audio settings, each audio file is split into two
                            audio_file_nr = math.floor(global_idx/2)
                            # if idx_mod_two == 0:

                            orig_audio = recovery.load_audio(cfg, audio_file_nr, which_half)

                            # orig_audio = audio_vgg.recover_audio(cfg, video)
                            writer.add_audio(orig_audio, tag="Original Input {}".format(global_idx))


                            if not cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.ENABLE:
                                # Permute to (T, H, W, C) from (C, T, H, W).


                                video = video.permute(1, 2, 3, 0)
                                video = data_utils.revert_tensor_normalize(
                                    video, cfg.DATA.MEAN, cfg.DATA.STD
                                )
                            else:
                                # Permute from (T, C, H, W) to (T, H, W, C)
                                video = video.permute(0, 2, 3, 1)

                            bboxes = (
                                None if cur_boxes is None else cur_boxes[:, 1:]
                            )
                            cur_prediction = (
                                cur_preds
                                if cfg.DETECTION.ENABLE
                                else cur_preds[cur_batch_idx]

                            )
                            video = video_vis.draw_clip(
                                video, cur_prediction
                                , bboxes=bboxes
                            )
                            torch.save(video, '/home/stureski/output_tensor_{}'.format(global_idx))
                            video = (
                                torch.from_numpy(np.array(video))
                                .permute(0, 3, 1, 2)
                                .unsqueeze(0)
                                #adds extra dimension in the beginning
                            )

                            # import pdb
                            # pdb.set_trace()
                            recovered_audio = recovery.iteratively_recover_audio(video, orig_audio)
                            # "video" should be a log-mel spectrogram with a GradCAM binary mask applied - resulting in only salient audio
                            #recovered_audio = audio_vgg.recover_audio(cfg, video)
                            # video_tensors.append(video)
                            # audio_tensors.append(recovered_audio)

                            writer.add_audio(recovered_audio, tag="Input {}/Pathway{}".format(global_idx, path_idx + 1))
                            if cfg.TENSORBOARD.MODEL_VIS.WAVEPLOT.ENABLE:
                                writer.add_waveplot(recovered_audio, tag="Input {}".format(global_idx))


                            writer.add_video(
                                #change this to print the name of the label- if not through the predicition, through a concurrent for loop
                                video,
                                tag="Input {}/Pathway {}".format(
                                    global_idx, path_idx + 1
                                ),
                            )
                            if cfg.TENSORBOARD.MODEL_VIS.WAVEPLOT.ENABLE:
                                writer.add_waveplot(recovered_audio, tag="Input {}".format(global_idx))



                    if cfg.TENSORBOARD.MODEL_VIS.ACTIVATIONS:

                        writer.plot_weights_and_activations(
                            cur_activations,
                            tag="Input {}/Activations: ".format(global_idx),
                            batch_idx=cur_batch_idx,
                            indexing_dict=indexing_dict,


                        )
            # import pdb
            # pdb.set_trace()


def perform_wrong_prediction_vis(vis_loader, model, cfg):
    """
    Visualize video inputs with wrong predictions on Tensorboard.
    Args:
        vis_loader (loader): video visualization loader.
        model (model): the video model to visualize.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    wrong_prediction_visualizer = WrongPredictionVis(cfg=cfg)
    for batch_idx, (inputs, labels, _, _) in tqdm.tqdm(enumerate(vis_loader)):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()

        # Some model modify the original input.
        inputs_clone = [inp.clone() for inp in inputs]

        preds = model(inputs)

        if cfg.NUM_GPUS > 1:
            preds, labels = du.all_gather([preds, labels])
            if isinstance(inputs_clone, (list,)):
                inputs_clone = du.all_gather(inputs_clone)
            else:
                inputs_clone = du.all_gather([inputs_clone])[0]

        if cfg.NUM_GPUS:
            # Transfer the data to the current CPU device.
            labels = labels.cpu()
            preds = preds.cpu()
            if isinstance(inputs_clone, (list,)):
                for i in range(len(inputs_clone)):
                    inputs_clone[i] = inputs_clone[i].cpu()
            else:
                inputs_clone = inputs_clone.cpu()

        # If using CPU (NUM_GPUS = 0), 1 represent 1 CPU.
        n_devices = max(cfg.NUM_GPUS, 1)
        for device_idx in range(1, n_devices + 1):
            wrong_prediction_visualizer.visualize_vid(
                video_input=inputs_clone,
                labels=labels,
                preds=preds.detach().clone(),
                batch_idx=device_idx * batch_idx,
            )

    logger.info(
        "Class indices with wrong predictions: {}".format(
            sorted(wrong_prediction_visualizer.wrong_class_prediction)
        )
    )
    wrong_prediction_visualizer.clean()


def visualize(cfg):
    """
    Perform layer weights and activations visualization on the model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    if cfg.TENSORBOARD.ENABLE and (
        cfg.TENSORBOARD.MODEL_VIS.ENABLE
        or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
    ):
        # Set up environment.
        du.init_distributed_training(cfg)
        # Set random seed from configs.
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)

        # Setup logging format.
        logging.setup_logging(cfg.OUTPUT_DIR)

        # Print config.
        logger.info("Model Visualization with config:")
        logger.info(cfg)
        # torch_log.getLogger('PIL').setLevel(torch_log.WARNING)

        # Build the video model and print model statistics.
        model = build_model(cfg)
        model.eval()
        # if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        #     misc.log_model_info(model, cfg, use_train_input=False)

        cu.load_test_checkpoint(cfg, model)

        # Create video testing loaders.
        vis_loader = loader.construct_loader(cfg, "test")

        #
        # if cfg.DETECTION.ENABLE:
        #     assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0

        logger.info("Testing model for {} iterations".format(len(vis_loader)))

        assert (
            len(vis_loader.dataset)
            % cfg.TEST.NUM_ENSEMBLE_VIEWS
            == 0
        )
        # Create meters for multi-view testing.
        if cfg.TEST.DATASET == 'epickitchens':
            test_meter = EPICTestMeter(
                len(vis_loader.dataset)
                // cfg.TEST.NUM_ENSEMBLE_VIEWS,
                cfg.TEST.NUM_ENSEMBLE_VIEWS,
                cfg.MODEL.NUM_CLASSES,
                len(vis_loader),
                cfg.DATA.ENSEMBLE_METHOD,
            )
        else:
            # test_meter = TestMeter(
            #     len(vis_loader.dataset)
            #     // cfg.TEST.NUM_ENSEMBLE_VIEWS,
            #     cfg.TEST.NUM_ENSEMBLE_VIEWS,
            #     cfg.MODEL.NUM_CLASSES[0],
            #     len(vis_loader),
            #     cfg.DATA.MULTI_LABEL,
            #     cfg.DATA.ENSEMBLE_METHOD,

            test_meter = TestMeter(
                len(vis_loader.dataset)
                // cfg.TEST.NUM_ENSEMBLE_VIEWS,
                cfg.TEST.NUM_ENSEMBLE_VIEWS,
                cfg.MODEL.NUM_CLASSES,
                len(vis_loader),
                cfg.DATA.MULTI_LABEL,
                cfg.DATA.ENSEMBLE_METHOD,
            )

        # Set up writer for logging to Tensorboard format.
        if du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
            writer = tb.TensorboardWriter(cfg)
        else:
            writer = None
        if cfg.TENSORBOARD.PREDICTIONS_PATH != "":
            assert not cfg.DETECTION.ENABLE, "Detection is not supported."
            logger.info(
                "Visualizing class-level performance from saved results..."
            )

            if writer is not None:
                with g_pathmgr.open(
                    cfg.TENSORBOARD.PREDICTIONS_PATH, "rb"
                ) as f:
                    preds, labels = pickle.load(f, encoding="latin1")

                writer.plot_eval(preds, labels)

        if cfg.TENSORBOARD.MODEL_VIS.ENABLE:
            if cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.ENABLE:
                assert (
                    not cfg.DETECTION.ENABLE
                ), "Detection task is currently not supported for Grad-CAM visualization."
                if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
                    assert (
                        len(cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.LAYER_LIST) == 1
                    ), "The number of chosen CNN layers must be equal to the number of pathway(s), given {} layer(s).".format(
                        len(cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.LAYER_LIST)
                    )
                elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
                    assert (
                        len(cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.LAYER_LIST) == 2
                    ), "The number of chosen CNN layers must be equal to the number of pathway(s), given {} layer(s).".format(
                        len(cfg.TENSORBOARD.MODEL_VIS.GRAD_CAM.LAYER_LIST)
                    )
                else:
                    raise NotImplementedError(
                        "Model arch {} is not in {}".format(
                            cfg.MODEL.ARCH,
                            cfg.MODEL.SINGLE_PATHWAY_ARCH
                            + cfg.MODEL.MULTI_PATHWAY_ARCH,
                        )
                    )
            logger.info(
                "Visualize model analysis for {} iterations".format(
                    len(vis_loader)
                )
            )
            # Run visualization on the model
            run_visualization(vis_loader, model, cfg, writer)
        if cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE:
            logger.info(
                "Visualize Wrong Predictions for {} iterations".format(
                    len(vis_loader)
                )
            )
            perform_wrong_prediction_vis(vis_loader, model, cfg)

        if writer is not None:
            writer.close()