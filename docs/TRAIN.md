# ProText Training

For training ProText, we have provided shell scripts in [scripts/protext](../scripts/protext) directory. Make sure to update the `DATAPATH` variable with dataset path in the script file and run the commands from the main directory `ProText/`.
Below we provide training and testing instructions for ProText.

### Training time, compute and log files
We train ProText on each dataset with a batch size of 256 text-to-text pairs using a **single** NVIDIA V100 16 GB GPU. Training ProText is lightweight and fast as we do not use image samples. Training ProText on ImageNet for 10 epochs takes around 20 minutes.  Additionally, to ease reproducing our main experimental results, we have provided training log files for [each dataset at this link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/muzammal_naseer_mbzuai_ac_ae/ErAYiYg5N0xKgJSDm-zeUKQB_6OAoIYb2giZvoLcAFoZSg?e=lxSEhb).

## ProText

#### (1) Base-to-Novel class generalization setting
The base-to-novel ProText config files have been provided at `configs/trainers/ProText/base2novel/` directory. Separate config files are present for each dataset, e.g `imagenet.yaml` should be used to train ProText on ImageNet-1k. All hyper-parameters such as text-data path, prompt length and prompt depth etc., can be modified using these config files.

Run the below shell script to train ProText on ImageNet base classes. This will also evaluate ProText on ImageNet novel classes.

```bash
# Other possible dataset values includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

# trains on base classes and then evaluate on both base and novel classes
bash scripts/protext/base2novel.sh imagenet output/experiment/path
```

This will produce results for both base and novel classes. The above steps can be repeated for other individual datasets.

#### (2) Cross-Dataset Transfer setting
We provide instructions to train ProText on ImageNet text-only data for all 1000 classes then evaluating it directly on new downstream datasets.
The corresponding cross-dataset config for ProText is available at: `configs/trainers/ProText/cross_datasets/imagenet.yaml`. All ProText hyper-parameters can be modified in this config file.

* Firstly, train ProText on imagenet source dataset.

```bash
# The second argument is the path for saving logs and model weights
bash scripts/protext/cross_datasets_train.sh imagenet output/imagenet_cross_dataset
```

* Now directly evaluate the ImageNet trained model on downstream cross-datasets.

```bash
# Other possible dataset values includes [imagenet, food101, dtd, ucf101, oxford_flowers, fgvc_aircraft, sun397, eurosat]
# The second argument is the folder path for the pretrained model weight
bash scripts/protext/cross_datasets_test.sh caltech101 output/imagenet_cross_dataset
bash scripts/protext/cross_datasets_test.sh oxford_pets output/imagenet_cross_dataset
bash scripts/protext/cross_datasets_test.sh stanford_cars output/imagenet_cross_dataset 
```

#### (3) Domain Generalization setting
We use ImageNet trained ProText model (trained for 200 epochs as compared to 10 epoch ImageNet used for cross-dataset transfer) for domain generalization experiments. The steps are similar to above cross-dataset experiments.

The corresponding domain generalization config for ProText is available at: `configs/trainers/ProText/text_only_supervised/imagenet.yaml`.

* Firstly, train ProText on imagenet source dataset.

```bash
# The second argument is the path for saving logs and model weights
bash scripts/protext/fully_supervised_and_dg.sh imagenet output/imagenet_dg
```
* Evaluate ImageNet model on different variants of ImageNet (datasets with domain shifts).

```bash
# The second argument is the folder path for the pretrained model weight
bash scripts/protext/dg_test.sh imagenetv2 output/imagenet_dg
bash scripts/protext/dg_test.sh imagenet_sketch output/imagenet_dg
bash scripts/protext/dg_test.sh imagenet_a output/imagenet_dg
bash scripts/protext/dg_test.sh imagenet_r output/imagenet_dg
```


#### (4) Text Only Supervised Setting
In this setting, ProText is trained on all classes individual datasets using their corresponding GPT LLM text data. The corresponding config files are available at `configs/trainers/ProText/text_only_supervised/` directory.

Now use the training script `scripts/protext/fully_supervised_and_dg.sh`. Run the commands below to calculate the results for imagenet dataset.

```bash
# Other possible dataset values includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

# train (text-only data) and test on given dataset 
bash scripts/protext/fully_supervised_and_dg.sh imagenet output/experiment/path
```
<br>

This repository also supports using official [PromptSRC](TRAIN_PromptSRC.md), [MaPLe](MaPLe.md), [CoOp](CoOp.md) and [Co-CoOp](Co-CoOp.md) configs and models.
