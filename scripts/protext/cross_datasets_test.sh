DATASET=$1
MODELPATH=$2
EXP=output_cross_dataset
DATAPATH=/path/to/datasets/folder

# Evaluate on cross-dataset
python train.py --root ${DATAPATH} --seed 1 --trainer ProText --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/ProText/cross_datasets/imagenet.yaml --output-dir ./${EXP}/${DATASET}_cross_dataset --eval-only --model-dir ${MODELPATH}