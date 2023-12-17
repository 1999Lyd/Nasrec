LR=0.16
WD=0

python -u dlrm_prune/structured_dlrm.py \
    --root_dir ./rec_data/criteo_kaggle_autoctr/ \
    --net dlrm \
    --supernet_config nasrec/configs/criteo/ea_criteo_kaggle_xlarge_best_1shot.json \
    --num_epochs 1\
    --learning_rate $LR \
    --train_batch_size 256\
    --wd $WD \
    --logging_dir ./experiments-www-repro/best_models/dlrm_%80_criteo_autoctr_best_1shot_lr${LR}_wd${WD} \
    --gpu 0 \
    --test_interval 10000