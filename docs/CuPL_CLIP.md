#### Benchmarking CuPL and CLIP baseline results on image datasets

As CuPL and CLIP Zero shot are training free methods, we provide script to evaluate these baselines on different datasets. Make sure to update the `DATAPATH` variable with dataset path in the script file and run the commands from the main directory `ProText/`.


#### CLIP Zero Shot Results

To reproduce CLIP Zeroshot inference results for image datasets, run the following command:

```bash
# Other possible dataset values includes [imagenet, food101, dtd, ucf101, oxford_flowers, fgvc_aircraft, sun397, eurosat]
# Perform zero shot inference of CLIP on ImageNet for all classes
bash scripts/zsclip_cupl/zeroshot.sh imagenet all None output/path/

# Perform zero shot inference of CLIP on ImageNet for base classes
bash scripts/zsclip_cupl/zeroshot.sh imagenet base None output/path/

# Perform zero shot inference of CLIP on ImageNet for novel classes
bash scripts/zsclip_cupl/zeroshot.sh imagenet new None output/path/

```

#### CuPL LLM based Ensembling Results

To reproduce CuPL inference results for image datasets, run the following command:

```bash
# Other possible dataset values includes [imagenet, food101, dtd, ucf101, oxford_flowers, fgvc_aircraft, sun397, eurosat]
# Perform CuPL inference on ImageNet for all classes
bash scripts/zsclip_cupl/zeroshot.sh imagenet all templates/CuPL_image_prompts.json output/path/

# Perform CuPL inference on ImageNet for base classes
bash scripts/zsclip_cupl/zeroshot.sh imagenet base templates/CuPL_image_prompts.json output/path/

# Perform CuPL inference on ImageNet for novel classes
bash scripts/zsclip_cupl/zeroshot.sh imagenet new templates/CuPL_image_prompts.json output/path/

```
Similarly for other datasets, LLM template path should be changed accordingly. For example: `templates/EuroSAT.json` should be used for the case of EuroSAT dataset. 

**Note:** For unseen class and dataset results for CuPL, we report same results as CLIP due to non-transferability of CuPL class-specific templates to new classes and datasets.