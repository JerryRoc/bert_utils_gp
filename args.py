# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:45:27 2019

@author: pg255026

define all parameters for example
"""
import os

file_path = os.path.dirname(__file__)
model_dir_name = "multi_cased_L-12_H-768_A-12" # 'uncased_L-12_H-768_A-12'
model_dir = os.path.join(file_path, model_dir_name)
config_name = os.path.join(model_dir, 'bert_config.json')
ckpt_name = os.path.join(model_dir, 'bert_model.ckpt')
output_dir = os.path.join(model_dir, '../tmp/result/')
tmp_model_name = os.path.join(output_dir, model_dir_name)
vocab_file = os.path.join(model_dir, 'vocab.txt')
data_dir = os.path.join(model_dir, '../data/')

num_train_epochs = 10
batch_size = 128
learning_rate = 0.00005
do_lower_case = True

# gpu使用率
gpu_memory_fraction = 0.8

# 默认取倒数四层的输出值作为句向量
layer_indexes = [-1,-2,-3,-4]

# 序列的最大程度，单文本建议把该值调小
max_seq_len = 32

# session-wise XLA doesn't seem to work on tf 1.10
xla = 0 #if tf.__version__ == '1.10.0' else 1