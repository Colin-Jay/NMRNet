dataset='crystal'
data_path="./demo/data/pretrain/${dataset}" # replace to your data path
dict_name="./demo/data/pretrain/${dataset}/dict.txt"  # replace to your dict path
n_gpu=1
MASTER_PORT=33333
lr=1e-4
wd=1e-4
batch_size=8
update_freq=2
masked_token_loss=1
masked_coord_loss=1
masked_dist_loss=1
dist_threshold=8.0
minkowski_p=2.0
lattice_loss=0
random_choice=0
x_norm_loss=0.01
delta_pair_repr_norm_loss=0.01
mask_prob=0.15
noise_type='uniform'
noise=1.0
seed=1
warmup_steps=120
max_steps=2000

lattice='0'   # 'atom3' 'atom6'   0   'cls3' 'cls6'
distance='global0' # 'global' 'unitcell' 
pretrain_weight='./weights/mol_pre_all_h_220816.pt'

global_batch_size=`expr $batch_size \* $n_gpu \* $update_freq`
save_dir="./demo/output/pretrain/${dataset}_distance_${distance}_lattice_${lattice}_lr_${lr}_mp_${mask_prob}_noise_${noise_type}_${noise}_bs_${global_batch_size}_loss_${masked_token_loss}_${masked_coord_loss}_${masked_dist_loss}_${minkowski_p}_${dist_threshold}_${lattice_loss}_${x_norm_loss}_${delta_pair_repr_norm_loss}_${warmup_steps}_${max_steps}"
if [ -d "${save_dir}" ]; then
    rm -rf ${save_dir}
    echo "Folder remove at: ${save_dir}"
fi
mkdir -p ${save_dir}
echo "Folder created at: ${save_dir}"
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --user-dir ./uninmr --train-subset train --valid-subset valid \
       --num-workers 16 --ddp-backend=c10d \
       --task unimat_rcut --loss unimat_rcut --arch unimol_large  \
       --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $wd \
       --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $max_steps \
       --update-freq $update_freq --seed $seed \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --tensorboard-logdir ${save_dir}/tsb \
       --max-update $max_steps --log-interval 100 --log-format simple \
       --save-interval-updates 200 --validate-interval-updates 200 --keep-interval-updates 10 --no-epoch-checkpoints  \
       --masked-token-loss $masked_token_loss --masked-coord-loss $masked_coord_loss --masked-dist-loss $masked_dist_loss \
       --x-norm-loss $x_norm_loss --delta-pair-repr-norm-loss $delta_pair_repr_norm_loss --lattice-loss $lattice_loss --random-choice $random_choice \
       --mask-prob $mask_prob --noise-type $noise_type --noise $noise --batch-size $batch_size --dist-threshold $dist_threshold --minkowski-p $minkowski_p \
       --required-batch-size-multiple 1  \
       --finetune-from-model $pretrain_weight --not-resample --dict-name $dict_name --gaussian-kernel --save-dir $save_dir  \
       --find-unused-parameters  \
> "${save_dir}/pretrain.log" 2>&1
# --remove-hydrogen
# --global-distance
