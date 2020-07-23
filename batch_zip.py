# -*- coding: utf-8 -*-
# @Time    : 2020/7/22 14:00
# @Author  : zhaohuayang
# @email   : zhaohuayang@myhexin.com
import os
import tensorflow as tf
from functional import seq
from multiprocessing.dummy import Pool as ThreadPool

model_names = [
    'albert_args_train_models/3_albert_xxlarge_v2_32_384_2e-05_2_0',
    'pv_ensemble_models/bs32_seq384_lr5e-05_ep2.0',
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
