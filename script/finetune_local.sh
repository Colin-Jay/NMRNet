dataset=$1  
data_path="/vepfs/fs_users/xufanjie/data/NMR/${dataset}" # replace to your data path
n_gpu=$2  
MASTER_PORT=33333
num_classes=1
weight_path='/vepfs/fs_ckps/xufanjie/UniNMR/saved'  # replace to your pre-training ckpt path
weight_name=$3
dict_path='/vepfs/fs_ckps/xufanjie/UniNMR/dict'
dict_name=$4
lr=$5  
batch_size=$6
epoch=$7  
dropout=0.0
warmup=0.06
update_freq=1

selected_atom=$8 
loss=$9
arch=${10}
if [ ${11} != 0 ]; then
    GLOBAL_DISTANCE_FLAG="--global-distance"
else
    GLOBAL_DISTANCE_FLAG=""
fi
if [ "${12}" = "gauss" ]; then
    GAUSS_FLAG="--gaussian-kernel"
else
    GAUSS_FLAG=""
fi

global_batch_size=`expr $batch_size \* $n_gpu \* $update_freq`
save_dir="/vepfs/fs_ckps/xufanjie/UniNMR/finetune/${dataset}/${selected_atom}_${weight_name}_global_${11}_kener_${12}_${arch}_${loss}_lr_${lr}_bs_${global_batch_size}_${warmup}_${epoch}"
if [ -d "${save_dir}" ]; then
    rm -rf ${save_dir}
    echo "Folder remove at: ${save_dir}"
fi
mkdir -p ${save_dir}
echo "Folder created at: ${save_dir}"
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --user-dir /root/NMRNet/uninmr  --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --task uninmr --loss $loss --arch $arch  \
       --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $batch_size \
       --update-freq $update_freq --seed 1 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --num-classes $num_classes --pooler-dropout $dropout \
       --finetune-from-model "${weight_path}/${weight_name}.pt" --dict-name "${dict_path}/${dict_name}.txt" \
       --log-interval 1000 --log-format simple \
       --validate-interval 1 --keep-last-epochs 1 \
       --save-dir $save_dir \
       --best-checkpoint-metric valid_rmse  \
       --selected-atom $selected_atom  --split-mode predefine $GLOBAL_DISTANCE_FLAG $GAUSS_FLAG \
> "${save_dir}/finetune.log" 2>&1

