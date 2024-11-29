data_path="/vepfs/fs_users/xufanjie/data/NMR/nmrshiftdb2/2024/All" # replace to your data path

save_dir="/vepfs/fs_ckps/xufanjie/UniNMR/finetune/shiftml1_rcut_filter/H/6/5cv/H_pretrain_csd_limit_rcut6_bs256_2_global_0_kener_gauss_unimol_large_atom_regloss_mse_lr_1e-4_bs_16_0.06_10"


python /vepfs/fs_users/xufanjie/NMRNet/uninmr/infer.py --user-dir  /vepfs/fs_users/xufanjie/NMRNet/uninmr   ${data_path}   --valid-subset valid \
    --results-path /vepfs/fs_users/xufanjie/data/NMR/nmrshiftdb2/2024_repair  --saved-dir $save_dir \
    --num-workers 8 --ddp-backend=c10d --batch-size 8 \
    --task uninmr --loss 'atom_regloss_mae' --arch 'unimol_large' \
    --dict-name "/vepfs/fs_ckps/xufanjie/UniNMR/dict/oc_limit_dict.txt" \
    --path ${save_dir}/cv_seed_${cv_seed}_fold_${fold}/checkpoint_best.pt \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
    --log-interval 50 --log-format simple --required-batch-size-multiple 1 \
    --selected-atom 'H'   --gaussian-kernel   --atom-descriptor 0 --split-mode infer 
done 2>&1 | tee /vepfs/fs_ckps/xufanjie/UniNMR/test/infer.log