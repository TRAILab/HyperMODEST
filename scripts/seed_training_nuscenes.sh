#!/bin/bash
set -e

formatstring="dynamic_%s_round_0"
base_dataset="nuscenes_boston"
model="pointrcnn_dynamic_obj"

while getopts "F:b:m:" opt
do
    case $opt in
        F) formatstring=$OPTARG ;;
        b) base_dataset=$OPTARG ;;
        m) model=$OPTARG ;;
        *)
            echo "there is unrecognized parameter."
            exit 1
            ;;
    esac
done

set -x
proj_root_dir=$(pwd)

iter_name=$(printf $formatstring 0)
if [ ! -d "${proj_root_dir}/downstream/OpenPCDet/data/nusc_${iter_name}" ]
then
    echo "=> Generating ${iter_name} dataset"
    cd ${proj_root_dir}/downstream/OpenPCDet/data
    mkdir nusc_${iter_name}
    cp -r ./${base_dataset}/training ./nusc_${iter_name}
    ln -s ${proj_root_dir}/downstream/OpenPCDet/data/${base_dataset}/ImageSets ./nusc_${iter_name}/
    ln -s ${proj_root_dir}/downstream/OpenPCDet/data/${base_dataset}/kitti_infos_val.pkl ./nusc_${iter_name}/
    cd ./nusc_${iter_name}/training
    if [ -L "label_2" ]; then
        rm label_2
    fi
    ln -s ${proj_root_dir}/generate_cluster_mask/intermediate_results/nusc_labels_pp_score_fw_30_r0.3_fov/ label_2
fi

# run data pre-processing
if [ ! -f "${proj_root_dir}/downstream/OpenPCDet/data/nusc_${iter_name}/.finish_tkn" ]
then
    echo "=> pre-processing ${iter_name} dataset"
    cd ${proj_root_dir}/downstream/OpenPCDet
    python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/nuscenes_boston_dynamic_obj.yaml ../data/nusc_${iter_name}
    touch ${proj_root_dir}/downstream/OpenPCDet/data/nusc_${iter_name}/.finish_tkn
fi

# start training
echo "=> ${iter_name} training"
cd ${proj_root_dir}/downstream/OpenPCDet/tools
bash scripts/dist_train.sh 2 --cfg_file cfgs/nuscenes_boston_models/${model}.yaml \
    --extra_tag ${iter_name} --merge_all_iters_to_one_epoch \
    --fix_random_seed \
    --set DATA_CONFIG.DATA_PATH ../data/nusc_${iter_name}

# generate the preditions on the training set
bash scripts/dist_test.sh 2 --cfg_file cfgs/nuscenes_boston_models/${model}.yaml \
    --extra_tag ${iter_name} --eval_tag trainset_0 \
    --ckpt ../output/nuscenes_boston_models/${model}/${iter_name}/ckpt/checkpoint_epoch_80.pth \
    --set DATA_CONFIG.DATA_PATH ../data/nusc_${iter_name} \
    DATA_CONFIG.DATA_SPLIT.test train DATA_CONFIG.INFO_PATH.test kitti_infos_train.pkl