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
from tensorflow.python.framework.graph_util import convert_variables_to_constants

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


def frozen_pb(checkpoint, export_path, tpu_address):
    output_pb = os.path.join(export_path, "frozen_model.pb")
    tf.reset_default_graph()

    saver = tf.train.import_meta_graph(checkpoint + '.meta', clear_devices=True)
    with tf.Session(tpu_address) as sess:
        saver.restore(sess, checkpoint)

        output_graph_def = convert_variables_to_constants(sess, sess.graph_def,
                                                          output_node_names=['answer_class/Squeeze',
                                                                             'start_logits/LogSoftmax',
                                                                             'end_logits/LogSoftmax'])
        with tf.gfile.FastGFile(output_pb, mode='wb') as f:
            f.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
    frozen_pb(FLAGS.checkpoint, FLAGS.export_path, FLAGS.tpu_address)
