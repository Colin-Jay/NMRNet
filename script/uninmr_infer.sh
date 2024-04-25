dataset=$1  
data_path="/vepfs/fs_users/xufanjie/data/NMR/${dataset}" # replace to your data path
dict_path='/vepfs/fs_ckps/xufanjie/UniNMR/dict'
dict_name=$2

saved_dir=$3
batch_size=8
selected_atom=$4
loss=$5
arch=$6
if [ $7 != 0 ]; then
    GLOBAL_DISTANCE_FLAG="--global-distance"
else
    GLOBAL_DISTANCE_FLAG=""
fi
if [ "$8" = "gauss" ]; then
    GAUSS_FLAG="--gaussian-kernel"
else
    GAUSS_FLAG=""
fi
exp=$(basename $saved_dir)
results_path=/vepfs/fs_ckps/xufanjie/UniNMR/infer/$dataset/$exp
if [ -d "${results_path}" ]; then
    rm -rf ${results_path}
    echo "Folder remove at: ${results_path}"
fi
mkdir -p ${results_path}
echo "Results dir:  $results_path"
echo "Start infering..."


python /vepfs/fs_users/xufanjie/Uni_NMR/Uni-Mol/uninmr/infer.py --user-dir /vepfs/fs_users/xufanjie/Uni_NMR/Uni-Mol/uninmr ${data_path}   --valid-subset valid \
        --results-path $results_path  --saved-dir $saved_dir \
        --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
        --task uninmr --loss atom_regloss_mse --arch unimol_large \
        --dict-name "${dict_path}/${dict_name}.txt" \
        --path ${saved_dir}/checkpoint_best.pt \
        --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
        --log-interval 50 --log-format simple --required-batch-size-multiple 1 \
        --selected-atom $selected_atom  --split-mode predefine $GLOBAL_DISTANCE_FLAG $GAUSS_FLAG \
        2>&1 | tee "${results_path}/infer.log"


