#!/usr/bin/env

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6061

det_weight=0.1
cls_weight=0.0005
num_bins=64
log_dir=./polyformer_b_reftask_logs_resume
save_dir=./polyformer_b_reftask_checkpoints_resume
mkdir -p $log_dir $save_dir

bpe_dir=../../utils/BPE
user_dir=../../polyformer_module

data_dir=../../datasets/finetune
data=${data_dir}/reftask_train_shuffled.tsv,${data_dir}/reftask/reftask_val.tsv
selected_cols=0,5,6,2,4,3,7
restore_file=../../pretrained_weights/polyformer_b_pretrain.pt

task=refcoco
arch=polyformer_b
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
lr=3e-5
max_epoch=5
warmup_ratio=0.06
batch_size=8
update_freq=8
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=80
max_tgt_length=420

patch_image_size=512

for max_epoch in 100; do
  echo "max_epoch "${max_epoch}
  for lr in 5e-5; do
    echo "lr "${lr}
    for patch_image_size in 512; do
      echo "patch_image_size "${patch_image_size}

      log_file=${log_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}".log"
      save_path=${save_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}
      mkdir -p $save_path

      CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} ../../train.py \
          $data \
          --selected-cols=${selected_cols} \
          --bpe-dir=${bpe_dir} \
          --user-dir=${user_dir} \
          --reset-optimizer --reset-dataloader --reset-meters \
          --save-dir=${save_path} \
          --task=${task} \
          --arch=${arch} \
          --criterion=${criterion} \
          --label-smoothing=${label_smoothing} \
          --batch-size=${batch_size} \
          --update-freq=${update_freq} \
          --encoder-normalize-before \
          --restore-file=${restore_file} \
          --decoder-normalize-before \
          --share-decoder-input-output-embed \
          --share-all-embeddings \
          --layernorm-embedding \
          --patch-layernorm-embedding \
          --code-layernorm-embedding \
          --resnet-drop-path-rate=${resnet_drop_path_rate} \
          --encoder-drop-path-rate=${encoder_drop_path_rate} \
          --decoder-drop-path-rate=${decoder_drop_path_rate} \
          --dropout=${dropout} \
          --attention-dropout=${attention_dropout} \
          --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=1.0 \
          --lr-scheduler=polynomial_decay --lr=${lr} \
          --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
          --log-format=simple --log-interval=10 \
          --fixed-validation-seed=7 \
          --no-epoch-checkpoints --keep-best-checkpoints=1 \
          --save-interval=1 --validate-interval=1 \
          --save-interval-updates=500 --validate-interval-updates=500 \
          --eval-acc \
          --eval-args='{"beam":5,"min_len":2,"max_len_a":0,"max_len_b":2}' \
          --best-checkpoint-metric=score --maximize-best-checkpoint-metric \
          --max-src-length=${max_src_length} \
          --max-tgt-length=${max_tgt_length} \
          --find-unused-parameters \
          --add-type-embedding \
          --scale-attn \
          --scale-fc \
          --scale-heads \
          --disable-entangle \
          --num-bins=${num_bins} \
          --patch-image-size=${patch_image_size} \
          --fp16 \
          --fp16-scale-window=512 \
          --det_weight=${det_weight} \
          --cls_weight=${cls_weight} \
          --num-workers=0 > ${log_file} 2>&1
    done
  done
done