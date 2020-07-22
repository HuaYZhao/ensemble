# -*- coding: utf-8 -*-
# @Time    : 2020/7/22 14:00
# @Author  : zhaohuayang
# @email   : zhaohuayang@myhexin.com
import os
import tensorflow as tf
from functional import seq
from multiprocessing.dummy import Pool as ThreadPool

model_names = [
    'args_train_models/1_electra_large_32_480_5e-05_2_1',
    'args_train_models/2_electra_large_32_384_5e-05_2_2',
    'args_train_models/2_electra_large_32_480_5e-05_2_2',
    'finetuning_models_atrlp/squad_model_1',
    'finetuning_models_atrlp/squad_model_5',
    'finetuning_models_atrlp/squad_model_9',
    'lr_epoch_models/3.0000000000000004e-05_2_3',
    'lr_epoch_models/6e-05_2_1',
    'lr_epoch_models/6e-05_3_1',
    'albert_args_train_models/2_albert_xxlarge_v1_32_384_2e-05_2_0',
    'albert_args_train_models/2_albert_xxlarge_v2_32_384_2e-05_2_0',
    'albert_args_train_models/3_albert_xlarge_v2_32_384_2e-05_2_0',
    'albert_args_train_models/3_albert_xxlarge_v1_32_384_2e-05_2_0',
    'albert_args_train_models/3_albert_xxlarge_v2_32_384_2e-05_2_0'
    'pv_ensemble_models/bs32_seq384_lr5e-05_ep2.0',
    'pv_ensemble_models/bs32_seq512_lr3e-05_ep3',
    'pv_ensemble_models/bs32_seq512_lr5e-05_ep2.0',
    'pv_ensemble_models/albert_xxlarge_2_384_2e-5',
]
models = (seq(tf.io.gfile.glob("gs://squad_cx/all_compressed_ensemble_models/*/*"))
          .filter(lambda x: any([y in x for y in model_names]))
          ).list()
print(models)

os.makedirs("my_ensemble_models")


def zip_a_model(model):
    xargs = f"gsutil -m cp -r {model} ./my_ensemble_models"
    os.system(xargs)

    model_name = os.path.basename(model)

    xargs = f"cd ./my_ensemble_models && 7z a -tzip {model_name}.zip {model_name} -r -mx=9"
    os.system(xargs)


if __name__ == '__main__':
    pool = ThreadPool(3)
    pool.map(zip_a_model, models)
