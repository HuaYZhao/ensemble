# -*- coding: utf-8 -*-
# @Time    : 2020/7/10 15:48
# @Author  : zhaohuayang
# @email   : zhaohuayang@myhexin.com
import os
import tensorflow as tf
import threading

TPU_NAMES = ['z1', 'z2', 'c1', ]


class myThread(threading.Thread):
    def __init__(self, tpu_id):
        threading.Thread.__init__(self)
        self.tpu_id = tpu_id

    def run(self):
        per_tpu_run(self.tpu_id)


def per_tpu_run(tpu_id):
    run_a_model(tpu_id, "electra_large", "large", 32, 512, 5e-5, 2)
    run_a_model(tpu_id, "electra_large", "large", 32, 384, 5e-5, 2)
    run_a_model(tpu_id, "electra_large", "large", 24, 512, 5e-5, 2)
    run_a_model(tpu_id, "electra_large", "large", 32, 480, 5e-5, 2)
    run_a_model(tpu_id, "electra_large", "large", 24, 480, 3e-5, 2)
    run_a_model(tpu_id, "electra_large", "large", 48, 256, 6e-5, 2)
    run_a_model(tpu_id, "electra_large", "large", 32, 512, 2e-5, 2)


def run_a_model(tpu_id, model_name, model_size, batch_size, max_seq_length, lr, epoch, run_time=0):
    print(f"Train model on {tpu_id}. args: model_name {model_name}, batch_size {batch_size}, "
          f"max_seq_length {max_seq_length}, lr {lr}, epoch {epoch}")

    xargs = f"""cd ../pv && python3 run_finetuning.py   --data-dir=gs://squad_cx/electra_data{tpu_id} --model-name={model_name}   --hparams '{{"model_size": "{model_size}", "task_names": ["squad"], "num_train_epochs": {epoch}, "use_tpu": true, "num_tpu_cores": 8, "tpu_name": "{TPU_NAMES[tpu_id - 1]}", "train_batch_size": {batch_size}, "eval_batch_size": 32, "predict_batch_size": 32, "max_seq_length": {max_seq_length}, "learning_rate": {lr}, "use_tfrecords_if_existing": false, "num_trials": 1, "do_train": true, "do_eval": true, "save_checkpoints_steps": 100000 }}' """
    os.system(xargs)

    xargs = f"gsutil -m cp -r gs://squad_cx/electra_data{tpu_id}/models/{model_name}/finetuning_models/squad_model_1 gs://squad_cx/args_train_pv_models/{tpu_id}_{model_name}_{batch_size}_{max_seq_length}_{lr}_{epoch}_{run_time}"
    os.system(xargs)

    xargs = f"gsutil -m cp -r gs://squad_cx/electra_data{tpu_id}/models/{model_name}/results/squad_qa gs://squad_cx/args_train_pv_results/{tpu_id}_{model_name}_{batch_size}_{max_seq_length}_{lr}_{epoch}_{run_time}"
    os.system(xargs)


if __name__ == '__main__':
    # 创建新线程
    thread1 = myThread(1)
    thread2 = myThread(2)
    thread3 = myThread(3)

    # 开启新线程
    thread1.start()
    thread2.start()
    thread3.start()

    thread1.join()
    thread2.join()
    thread3.join()
    print("退出主线程")
