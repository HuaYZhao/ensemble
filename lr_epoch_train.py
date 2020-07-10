# -*- coding: utf-8 -*-
# @Time    : 2020/7/9 9:27
# @Author  : zhaohuayang
# @email   : zhaohuayang@myhexin.com
import os
import numpy as np
import threading

TPU_NAMES = ['z1', 'z2', 'c1', ]

LrRange = np.arange(1e-5, 1e-4 + 1e-5, 1e-5)

EpochRange = [2, 3]


class myThread(threading.Thread):
    def __init__(self, tpu_id):
        threading.Thread.__init__(self)
        self.tpu_id = tpu_id

    def run(self):
        per_tpu_run(self.tpu_id)


def per_tpu_run(tpu_id):
    for lr in LrRange:
        for epoch in EpochRange:
            print(f"train lr: {lr}, epoch: {epoch}, tpu: {TPU_NAMES[tpu_id - 1]}")

            xargs = f"""cd ../atrlp && python3 run_finetuning.py   --data-dir=gs://squad_cx/electra_data{tpu_id} --model-name=electra_large   --hparams '{{"model_size": "large", "task_names": ["squad"], "num_train_epochs": {epoch}, "use_tpu": true, "num_tpu_cores": 8, "tpu_name": "{TPU_NAMES[tpu_id - 1]}", "train_batch_size": 32, "eval_batch_size": 32, "predict_batch_size": 32, "max_seq_length": 512, "learning_rate": {lr}, "use_tfrecords_if_existing": true, "num_trials": 1, "do_train": true, "do_eval": true, "save_checkpoints_steps": 100000 }}' """
            os.system(xargs)

            xargs = f"gsutil -m cp -r gs://squad_cx/electra_data{tpu_id}/models/electra_large/finetuning_models/squad_model_1 gs://squad_cx/lr_epoch_models/{lr}_{epoch}_{tpu_id}"
            os.system(xargs)

            xargs = f"gsutil -m cp -r gs://squad_cx/electra_data{tpu_id}/models/electra_large/results/squad_qa gs://squad_cx/lr_epoch_results/{lr}_{epoch}_{tpu_id}"
            os.system(xargs)


if __name__ == '__main__':
    # 创建新线程
    thread1 = myThread(1, )
    thread2 = myThread(2, )
    thread3 = myThread(3, )
    # 开启新线程
    thread1.start()
    thread2.start()
    thread3.start()

    thread1.join()
    thread2.join()
    thread3.join()
    print("退出主线程")
