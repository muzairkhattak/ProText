DATASET=$1
EXP=$2
DATAPATH=/path/to/datasets/folder

# Train on all classes for a given dataset
# --seed is only a place holder
python train.py --root ${DATAPATH} --seed 1 --trainer ProText --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/ProText/cross_datasets/imagenet.yaml --output-dir ${EXP}
