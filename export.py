#!/usr/bin/env python
# encoding:utf-8
# -----------------------------------------#
# Filename:     export.py
#
# Description:  delete useless nodes in inference
# Version:      1.0
# Created:      2020/3/20 15:46
# Author:       chenxiang@myhexin.com
# Company:      www.iwencai.com
#
# -----------------------------------------#
import os
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.contrib.framework import list_variables, load_variable

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint", None, "checkpoint path")
flags.DEFINE_string("export_path", None, "output node names")
flags.DEFINE_string("tpu_address", None, "tpu address")


def export(checkpoint, export_path, tpu_address):
    output_ckpt = os.path.join(export_path, "model.ckpt-best")
    tf.reset_default_graph()
    clear_devices = True
    saver = tf.train.import_meta_graph(checkpoint + '.meta', clear_devices=clear_devices)
    with tf.Session(tpu_address) as sess:
        saver.restore(sess, checkpoint)
        my_vars = []
        for var in tf.global_variables():
            if "adam_v" not in var.name and "adam_m" not in var.name:
                my_vars.append(var)
        new_saver = tf.train.Saver(my_vars)
        new_saver.save(sess, output_ckpt)


if __name__ == '__main__':
    export(FLAGS.checkpoint, FLAGS.export_path, FLAGS.tpu_address)
