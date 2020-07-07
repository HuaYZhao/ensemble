# -*- coding: utf-8 -*-
# @Time    : 2020/7/7 14:53
# @Author  : zhaohuayang
# @email   : zhaohuayang@myhexin.com
import os
from data_sampling import generate_bootstrap_train_data
import _thread

TPU_NAMES = ['z1', 'z2', 'c1', ]

per_tpu_run_times = 13
bootstrap_times = len(TPU_NAMES) * per_tpu_run_times

# generate_bootstrap_train_data(bootstrap_times)


def per_tpu_run(tpu_id, train_data_range):
    for i in train_data_range:
        print(f"train bootstrap {i}")
        xargs = f"gsutil cp bootstrap_train_{i}.json gs://squad_cx/electra_data{tpu_id}/finetuning_data/squad/train.json"
        os.system(xargs)

        xargs = f"""cd ../atrlp && python3 run_finetuning.py   --data-dir=gs://squad_cx/electra_data{tpu_id} --model-name=electra_large   --hparams '{"model_size": "large", "task_names": ["squad"], "num_train_epochs": 2, "use_tpu": true, "num_tpu_cores": 8, "tpu_name": "{TPU_NAMES[tpu_id - 1]}", "train_batch_size": 32, "eval_batch_size": 32, "predict_batch_size": 32, "max_seq_length": 512, "learning_rate": 5e-05, "use_tfrecords_if_existing": false, "num_trials": 1, "do_train": true, "do_eval": true, "save_checkpoints_steps": 100000 }' """
        os.system(xargs)

        xargs = f"gsutil -m cp -r gs://squad_cx/electra_data{tpu_id}/models/electra_large/finetuning_models/squad_model_1 gs://squad_cx/bootstrap_models/{i}"
        os.system(xargs)

        xargs = f"gsutil -m cp -r gs://squad_cx/electra_data{tpu_id}/models/electra_large/results/squad_qa gs://squad_cx/bootstrap_results/{i}"
        os.system(xargs)


if __name__ == '__main__':
    _thread.start_new_thread(per_tpu_run, (1, range(13),))
    _thread.start_new_thread(per_tpu_run, (2, range(13, 26),))
    _thread.start_new_thread(per_tpu_run, (3, range(26, 39),))
