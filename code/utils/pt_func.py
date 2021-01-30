#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/16 19:45
# @Author  : QXTD-LXH
# @Desc    :
import os
import torch



def load_tf_weights_in_bert(model, tf_checkpoint_path, strip_bert=False):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        print(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    # logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        # logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    def _safe_load(_p, _w_name):
        try:
            _p = getattr(_p, _w_name)
        except AttributeError:
            _p = None
        return _p

    for name, array in zip(names, arrays):
        if strip_bert:
            name = name[5:] if name.startswith('bert/') else name
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
                n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
                for n in name
        ):
            # logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = _safe_load(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = _safe_load(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = _safe_load(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = _safe_load(pointer, "classifier")
            else:
                pointer = _safe_load(pointer, scope_names[0])
            if pointer is None:
                break
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                try:
                    pointer = pointer[num]
                except IndexError:
                    pointer = None
                    break
        if pointer is None:
            print('Skip ', name)
            continue
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            if pointer.shape != array.shape:
                # print(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
                print('shape mis_math ', name)
                continue
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        # logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model
