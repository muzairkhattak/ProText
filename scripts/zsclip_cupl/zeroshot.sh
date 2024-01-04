
DATASET=$1
SAMPLE=$2
CUPL=$3
EXP=output_zs_evaluation
DATAPATH=/path/to/datasets/folder
# Zeroshot CLIP evaluation
# --seed is only a place holder
python train.py --root ${DATAPATH} --seed 1 --trainer ZeroshotCLIP --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/ProText/text_only_supervised/imagenet.yaml --output-dir ./${EXP}/${DATASET}_${SAMPLE}_${CUPL} --eval-only DATASET.SUBSAMPLE_CLASSES ${SAMPLE} TRAINER.PROTEXT.GPT_PATH ${CUPL}
