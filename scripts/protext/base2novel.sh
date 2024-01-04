DATASET=$1
EXP=$2
DATAPATH=/path/to/datasets/folder

# Train and test ProText on base classes
# --seed is only a place holder
python train.py --root ${DATAPATH} --seed 1 --trainer ProText --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/ProText/base2novel/${DATASET}.yaml --output-dir ${EXP} DATASET.SUBSAMPLE_CLASSES base

# Now also perform evaluation on novel classes
python train.py --root ${DATAPATH} --seed 1 --trainer ProText --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/ProText/base2novel/${DATASET}.yaml --output-dir ${DATASET}_novel --eval-only --model-dir ${EXP} DATASET.SUBSAMPLE_CLASSES new