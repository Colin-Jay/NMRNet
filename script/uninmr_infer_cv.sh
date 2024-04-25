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
# results_path=/vepfs/fs_ckps/xufanjie/UniNMR/infer/$dataset/5cv/$exp
# if [ -d "${results_path}" ]; then
#     rm -rf ${results_path}
#     echo "Folder remove at: ${results_path}"
# fi
# mkdir -p ${results_path}
# echo "Results dir:  $results_path"
nfolds=5
# for ((fold=0;fold<$nfolds;fold++))
for fold in $(seq 0 $(($nfolds - 1)))
    do
    echo "Start infering..."
    cv_seed=42
    python /vepfs/fs_users/xufanjie/Uni_NMR/Uni-Mol/uninmr/infer.py --user-dir /vepfs/fs_users/xufanjie/Uni_NMR/Uni-Mol/uninmr ${data_path}   --valid-subset valid \
        --results-path $saved_dir/cv_seed_${cv_seed}_fold_${fold}  --saved-dir $saved_dir \
        --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
        --task uninmr --loss $loss --arch $arch \
        --dict-name "${dict_path}/${dict_name}.txt" \
        --path ${saved_dir}/cv_seed_${cv_seed}_fold_${fold}/checkpoint_best.pt \
        --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
        --log-interval 50 --log-format simple --required-batch-size-multiple 1 \
        --selected-atom $selected_atom  $GLOBAL_DISTANCE_FLAG $GAUSS_FLAG    
    done 2>&1 | tee "${saved_dir}/infer.log"



python /vepfs/fs_users/xufanjie/Uni_NMR/Uni-Mol/uninmr/utils/get_result.py --path $saved_dir  --mode cv  2>&1 | tee ${saved_dir}/result.log