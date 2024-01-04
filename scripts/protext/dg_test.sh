DATASET=$1
MODELPATH=$2
EXP=output_domain_generalization
DATAPATH=/path/to/datasets/folder

# Evaluate on cross-dataset
python train.py --root ${DATAPATH} --seed 1 --trainer ProText --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/ProText/text_only_supervised/imagenet.yaml --output-dir ./${EXP}/${DATASET}_domain_generalization --eval-only --model-dir ${MODELPATH}