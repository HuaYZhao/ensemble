# -*- coding: utf-8 -*-
# @Time    : 2020/7/22 9:10
# @Author  : zhaohuayang
# @email   : zhaohuayang@myhexin.com
import os
import tensorflow as tf
import threading
from functional import seq

TPU_NAMES = ['z1', 'z2', 'c1', 'c2', 'c3']


class myThread(threading.Thread):
    def __init__(self, tpu_id, tasks):
        threading.Thread.__init__(self)
        self.tpu_id = tpu_id
        self.tasks = tasks

    def run(self):
        per_tpu_run(self.tpu_id, self.tasks)


def per_tpu_run(tpu_id, tasks):
    for task in tasks:
        pack_a_model(tpu_id, task)


def pack_a_model(tpu_id, model_dir):
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)

    output_dir = model_dir.replace("all_ensemble_models", "all_compressed_ensemble_models")

    xargs = f"python3 export.py --checkpoint {latest_checkpoint} --export_path {output_dir} --tpu_address grpc://10.121.47.58:8470"
    os.system(xargs)

    if "albert" in model_dir:
        if "xlarge" in model_dir:
            pretrain_dir = "gs://squad_cx/albert_data/pretrain_models/albert_xlarge_v1"
        elif "xxlarge" in model_dir:
            pretrain_dir = "gs://squad_cx/albert_data/pretrain_models/albert_xxlarge_v1"
        else:
            raise
        xargs = f"gsutil cp {pretrain_dir}/30k-clean.model {pretrain_dir}/30k-clean.vocab {pretrain_dir}/albert_config.json {output_dir}"
        os.system(xargs)
    else:
        xargs = f"gsutil cp gs://squad_cx/electra_data/models/electra_large/vocab.txt {output_dir}"
        os.system(xargs)

    print(f"finish packing {model_dir}")


if __name__ == '__main__':
    # all_models = (seq(tf.io.gfile.glob("gs://squad_cx/all_ensemble_models/*/*"))
    #               .filter_not(lambda x: "base" in x)
    #               .filter_not(lambda x: "albert" in x and "large" in x)
    #               ).list()
    all_models = [
        'gs://squad_cx/all_ensemble_models/albert_args_train_answer_models/1_albert_xlarge_v1_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/albert_args_train_answer_models/1_albert_xlarge_v2_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/albert_args_train_answer_models/1_albert_xxlarge_v1_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/albert_args_train_answer_models/1_albert_xxlarge_v2_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/albert_args_train_answer_models/2_albert_xlarge_v1_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/albert_args_train_answer_models/2_albert_xlarge_v2_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/albert_args_train_answer_models/2_albert_xxlarge_v1_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/albert_args_train_answer_models/2_albert_xxlarge_v2_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/albert_args_train_answer_models/3_albert_xlarge_v1_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/albert_args_train_answer_models/3_albert_xlarge_v2_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/albert_args_train_answer_models/3_albert_xxlarge_v1_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/albert_args_train_answer_models/3_albert_xxlarge_v2_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/albert_args_train_models/1_albert_xlarge_v1_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/albert_args_train_models/1_albert_xlarge_v2_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/albert_args_train_models/1_albert_xxlarge_v1_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/albert_args_train_models/1_albert_xxlarge_v2_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/albert_args_train_models/2_albert_xlarge_v1_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/albert_args_train_models/2_albert_xlarge_v2_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/albert_args_train_models/2_albert_xxlarge_v1_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/albert_args_train_models/2_albert_xxlarge_v2_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/albert_args_train_models/3_albert_xlarge_v1_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/albert_args_train_models/3_albert_xlarge_v2_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/albert_args_train_models/3_albert_xxlarge_v1_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/albert_args_train_models/3_albert_xxlarge_v2_32_384_2e-05_2_0',
        'gs://squad_cx/all_ensemble_models/pv_ensemble_models/albert_xxlarge_2_384_2e-5',
        'gs://squad_cx/all_ensemble_models/pv_ensemble_models/bs32_seq384_lr5e-05_ep2.0',
        'gs://squad_cx/all_ensemble_models/pv_ensemble_models/bs32_seq512_lr3e-05_ep3',
        'gs://squad_cx/all_ensemble_models/pv_ensemble_models/bs32_seq512_lr5e-05_ep2.0']

    num = len(all_models) // len(TPU_NAMES) + 1
    threads = []
    for i in range(len(TPU_NAMES)):
        threads.append(
            myThread(i + 1, all_models[num * i:num * (i + 1)])
        )
    for thred in threads:
        thred.start()
    for thred in threads:
        thred.join()
    print("退出主线程")
