# -*- coding: utf-8 -*-
# @Time    : 2020/7/13 19:22
# @Author  : zhaohuayang
# @email   : zhaohuayang@myhexin.com
import os
import threading
import shutil

TPU_NAMES = ['z1', 'z2', 'c1', ]


class myThread(threading.Thread):
    def __init__(self, tpu_id):
        threading.Thread.__init__(self)
        self.tpu_id = tpu_id

    def run(self):
        per_tpu_run(self.tpu_id)


def per_tpu_run(tpu_id):
    run_a_model(tpu_id, "albert_base_v1")
    run_a_model(tpu_id, "albert_base_v2")
    run_a_model(tpu_id, "albert_large_v1")
    run_a_model(tpu_id, "albert_large_v2")
    run_a_model(tpu_id, "albert_xlarge_v1")
    run_a_model(tpu_id, "albert_xlarge_v2")
    run_a_model(tpu_id, "albert_xxlarge_v1")
    run_a_model(tpu_id, "albert_xxlarge_v2")


def run_a_model(tpu_id, model_type, batch_size=32, max_seq_length=384, lr=2e-5, epoch=2, run_time=0):
    config_file = f"gs://squad_cx/albert_data/pretrain_models/{model_type}/albert_config.json"
    output_dir = f"gs://squad_cx/albert_data/train/{tpu_id}_{model_type}_{batch_size}_{max_seq_length}_{lr}_{epoch}_{run_time}"
    train_file = f"gs://squad_cx/albert_data/inputs/train-v2.0.json"
    predict_file = f"gs://squad_cx/albert_data/inputs/dev-v2.0.json"
    train_feature_file = f"gs://squad_cx/albert_data/features/train_features_{max_seq_length}_128_64"
    predict_feature_file = f"gs://squad_cx/albert_data/features/predict_features_{max_seq_length}_128_64"
    predict_feature_left_file = f"gs://squad_cx/albert_data/features/predict_features_left_{max_seq_length}_128_64"
    init_checkpoint = f"gs://squad_cx/albert_data/pretrain_models/{model_type}/model.ckpt-best"
    spm_model_file = f"gs://squad_cx/albert_data/pretrain_models/{model_type}/30k-clean.model"
    spm_vocab = f"gs://squad_cx/albert_data/pretrain_models/{model_type}/30k-clean.vocab"
    xargs = f"gsutil cp {spm_model_file} {spm_vocab} ./"
    os.system(xargs)
    spm_model = os.path.join(os.path.dirname(__file__), "30k-clean.model")

    xargs = f""" cd ../ALBERT && \
            python3 run_squad_v2.py \
              --albert_config_file={config_file} \
              --output_dir={output_dir} \
              --train_file={train_file} \
              --predict_file={predict_file} \
              --train_feature_file={train_feature_file} \
              --predict_feature_file={predict_feature_file} \
              --predict_feature_left_file={predict_feature_left_file} \
              --init_checkpoint={init_checkpoint} \
              --spm_model_file={spm_model} \
              --max_seq_length={max_seq_length} \
              --do_train=True \
              --do_predict=True \
              --train_batch_size={batch_size} \
              --predict_batch_size=32 \
              --learning_rate={lr} \
              --num_train_epochs={epoch} \
              --save_checkpoints_steps=100000 \
              --n_best_size=20 \
              --use_tpu=True \
              --tpu_name={TPU_NAMES[tpu_id - 1]}
            """
    os.system(xargs)

    xargs = f"gsutil -m cp -r gs://squad_cx/albert_data/train/{tpu_id}_{model_type}_{batch_size}_{max_seq_length}_{lr}_{epoch}_{run_time} gs://squad_cx/albert_args_train_models"
    os.system(xargs)

    os.mkdir(f'{tpu_id}_{model_type}_{batch_size}_{max_seq_length}_{lr}_{epoch}_{run_time}')
    xargs = f"gsutil cp gs://squad_cx/albert_data/train/{tpu_id}_{model_type}_{batch_size}_{max_seq_length}_{lr}_{epoch}_{run_time}/null_odds.json {tpu_id}_{model_type}_{batch_size}_{max_seq_length}_{lr}_{epoch}_{run_time}/squad_null_odds.json"
    os.system(xargs)
    xargs = f"gsutil cp gs://squad_cx/albert_data/train/{tpu_id}_{model_type}_{batch_size}_{max_seq_length}_{lr}_{epoch}_{run_time}/predictions.json {tpu_id}_{model_type}_{batch_size}_{max_seq_length}_{lr}_{epoch}_{run_time}/squad_preds.json"
    os.system(xargs)
    xargs = f"gsutil cp gs://squad_cx/albert_data/train/{tpu_id}_{model_type}_{batch_size}_{max_seq_length}_{lr}_{epoch}_{run_time}/eval_all_nbest.pkl {tpu_id}_{model_type}_{batch_size}_{max_seq_length}_{lr}_{epoch}_{run_time}/eval_all_nbest.pkl"
    os.system(xargs)
    xargs = f"gsutil -m cp -r {tpu_id}_{model_type}_{batch_size}_{max_seq_length}_{lr}_{epoch}_{run_time} gs://squad_cx/albert_args_train_results/"
    os.system(xargs)

    shutil.rmtree(f'{tpu_id}_{model_type}_{batch_size}_{max_seq_length}_{lr}_{epoch}_{run_time}')


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
