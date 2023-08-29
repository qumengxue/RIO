#!/bin/bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6092
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4


########################## Evaluate Refcoco+ ##########################
user_dir=../../polyformer_module
bpe_dir=../../utils/BPE
selected_cols=0,5,6,2,4,3


model='polyformer_b'
num_bins=64
batch_size=64

dataset='reftask'
ckpt_path=/path/to/checkpoint.pt


# split='test'
split='val'
data_dir=../../datasets/finetune/reftask
data=${data_dir}/reftask_${split}.tsv
# data=${data_dir}/reftask_test.tsv
result_path=../../results_${model}/${dataset}/
vis_dir=${result_path}/vis/${split}
result_dir=${result_path}/result/${split}
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${ckpt_path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=${batch_size} \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --num-bins=${num_bins} \
    --vis \
    --vis_dir=${vis_dir} \
    --result_dir=${result_dir} \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"
done