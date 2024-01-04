# Evaluating and Reproducing ProText Results

Below we provide instructions to reproduce ProText main experimental results using our pre-trained models. We use bash scripts in [scripts/protext](../scripts/protext) directory for evaluating ProText using the provided pre-trained model checkpoints. 

Make sure to update the `DATAPATH` variable with dataset path in the script file and run the commands from the main directory `ProText/`. 

## ProText

#### (1) Base-to-Novel class generalization setting
The base-to-novel ProText config files have been provided at `configs/trainers/ProText/base2novel/` directory. Separate config files are present for each dataset, e.g `imagenet.yaml` should be used to train ProText on ImageNet-1k. All hyper-parameters such as text-data path, prompt length and prompt depth etc., can be modified using these config files. No hyper-parameters or other settings should be changed in the config file during evaluation of pre-trained models. 

We show an example to reproduce results for imagenet. Follow the instructions below to reproduce results using our pre-trained model weights:
* Download the folder containing base-to-novel generalization pre-trained weights for a single dataset from this [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/muzammal_naseer_mbzuai_ac_ae/ElE2Dn32FqNHoliuTJULmiwBVxL1xMIE5sfdhEFOyeZwuA?e=Li7z7k). After downloading, the directory should look like this:

```
imagenet_base
|–– tensorboard/
|–– VLPromptLearner/
|–– log.txt
```

Now use the script `scripts/protext/base2novel.sh` and run the command below to calculate the results:
```bash
# Other possible dataset values includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

# evaluate on both base and novel classes using pretrained weights
bash scripts/protext/base2novel.sh imagenet path/to/imagenet_base
```

This should evaluate and show results for imagenet in base-to-novel class generalization setting.

#### (2) Cross-dataset setting
In cross-dataset, we first train ProText on ImageNet-1k and then evaluate the trained model directly on cross-datasets.

We provide the instructions below to reproduce cross-datasets and domain generalization results using ProText pre-trained models:
* Download the folder containing pre-trained weights for imagenet from this [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/muzammal_naseer_mbzuai_ac_ae/EqRdccAsKuRNsv1SOGZISI4BHCDTOzgMPWJlZK0hAL-Ylg?e=ZsAhBh). After downloading, the directory should look like this:

```
imagenet
|–– tensorboard/
|–– VLPromptLearner/
|–– log.txt
```

Now use the script `scripts/protext/cross_datasets_test.sh` and run the commands below to calculate the results for food101 dataset over 3 seeds:
```bash
# Other possible dataset values for cross-datasets includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

# evaluate on given dataset for SEED1
bash scripts/protext/cross_datasets_test.sh food101 /path/to/imagenet/folder
```

This should evaluate cross-dataset transfer results on ImageNet and save the log files in `output_cross_dataset/` directory.
The same above steps can be repeated for other individual datasets by providing respective dataset name and checkpoints path.


#### (3) Domain generalization setting
Domain generalization setting evaluation is similar to Cross-Dataset but we use ImageNet-1k ProText model trained with 200 epochs and context vectors.

We provide the instructions below to reproduce domain generalization results using our pre-trained imagenet model weights:
* Download the folder containing pre-trained weights for imagenet from this [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/muzammal_naseer_mbzuai_ac_ae/ElEJ6F82BCNGt3pksDguNm0BxELO7mAsfBTtSXyZGOtnHQ?e=nVUs3S). 

Now use the evaluation script `scripts/protext/dg_test.sh` and run the commands below to calculate the results for out of distribution datasets:
```bash
# possible dataset values for domain generalization benchmark includes [imagenetv2, imagenet_sketch, imagenet_a, imagenet_r]

bash scripts/protext/dg_test.sh imagenet_a /path/to/imagenet/weights/folder
bash scripts/protext/dg_test.sh imagenet_r /path/to/imagenet/weights/folder
bash scripts/protext/dg_test.sh imagenetv2 /path/to/imagenet/weights/folder
bash scripts/protext/dg_test.sh imagenet_sketch /path/to/imagenet/weights/folder

```

This should evaluate and save the log files in `output_domain_generalization/` directory. To obtain the results averaged over 3 seeds, run:


#### (4) Text Only Supervised Setting
In this setting, ProText is trained on all classes individual datasets using their corresponding text-only datasets. The config files are available at: `configs/trainers/ProText/text_only_supervised`. 
Follow the instructions below to reproduce results in this setting using our pre-trained models:

* Download the folder containing pre-trained weights for all food101 from this [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/muzammal_naseer_mbzuai_ac_ae/Eod0PjkCQaxJsu5q7B7FIXwBBTj9_ZYnll9R-BviqJ6BvA?e=VZhdjD).

Now use the script `scripts/protext/fully_supervised_and_dg.sh` and run the command below to reproduce ProText result for food101 dataset:
```bash
# Other possible dataset values includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

# evaluate on given dataset for K=1 shot
bash scripts/protext/fully_supervised_and_dg.sh food101 /path/to/food101/weights/folder
```

This should evaluate and print out the results. 
The same above steps can be repeated for other individual datasets by providing respective dataset name and checkpoints path.

<br>

This repository also supports using official [PromptSRC](EVAL_PromptSRC.md), [MaPLe](MaPLe.md), [CoOp](CoOp.md) and [Co-CoOp](Co-CoOp.md) configs and models.

## CuPL and ZS CLIP

Please refer to [CuPL_CLIP.md](CuPL_CLIP.md) for reproducing results of CuPL and CLIP methods. 