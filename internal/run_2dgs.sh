SCENE_DIR="/home/luzhan/Projects/scene_gen/consistent_3dscene/stable-virtual-camera/work_dirs/demo/img2trajvid_s-prob"
RESULT_DIR="results_meta/3dscene_with_intrinsics"
SCENE_LIST="vasedeck/112_views/dense"

for SCENE in $SCENE_LIST;
do
    DATA_FACTOR=1
    echo "Running $SCENE"

    # train without eval
    CUDA_VISIBLE_DEVICES=0 python train_3dscene_2dgs.py --eval_steps -1 --disable_viewer --data_factor $DATA_FACTOR \
        --model_type 2dgs \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/ \
        --max_steps 30000 \
        --intrinsics_loss \
        --intrinsics_lambda 1

    # run eval and render
    for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    do
        CUDA_VISIBLE_DEVICES=0 python train_3dscene_2dgs.py --disable_viewer --data_factor $DATA_FACTOR \
            --model_type 2dgs \
            --data_dir $SCENE_DIR/$SCENE/ \
            --result_dir $RESULT_DIR/$SCENE/ \
            --ckpt $CKPT \
            --intrinsics_loss \
            --grow_grad2d 0.0004
    done
done