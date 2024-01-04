import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import convert_weights
import json
from .coop import load_clip_to_cpu
# from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
from templates.imagenet_templates import IMAGENET_TEMPLATES
from templates.mapper_data import ctx_templates

CUSTOM_TEMPLATES_PROTEXT = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "a photo of a {}, a type of a texture.",
    "EuroSAT": "a photo of a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of a {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Original classnames: {classnames}")
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        use_gpt_prompts = cfg.TRAINER.PROTEXT.GPT_PATH
        use_attribute_prompts = cfg.TRAINER.PROTEXT.USE_ATTRIBUTE_DATA
        use_80_prompts = cfg.TRAINER.PROTEXT.USE_TEMPLATES
        mean_text_features = 0
        if use_80_prompts:
            print("Using standard 80 openai templates for text embeddings")
            total_templates_to_use = IMAGENET_TEMPLATES
            print(f"Prompt ensembling (n={len(total_templates_to_use)})")
            for i, temp in enumerate(total_templates_to_use):
                prompts = [temp.format(c.replace("_", " ")) for c in classnames]
                prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
                with torch.no_grad():
                    text_features = clip_model.encode_text(prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                mean_text_features = mean_text_features + text_features
            mean_text_features = mean_text_features / len(total_templates_to_use)
            mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)
            text_features = mean_text_features
        if use_gpt_prompts != None:
            print("Using CuPL templates for text embeddings")
            old_gpt_all = []
            file = open(use_gpt_prompts, "r")
            GPT_prompt_dict = json.load(file)
            # The order of embeddings should follow strictly order of classname variable
            # Keys name should match classnames so that we could do fetching from the dict.
            # Convert the dict to lower case
            GPT_prompt_dict = {k.lower().replace("_", " "): v for k, v in GPT_prompt_dict.items()}
            k = 0
            for single_key in classnames:
                single_class_prompts = GPT_prompt_dict[single_key.lower().replace("_", " ")]
                k += 1
                x_tokenized = torch.cat([clip.tokenize(p) for p in single_class_prompts])
                with torch.no_grad():
                    text_features = clip_model.encode_text(x_tokenized.cuda())
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                old_gpt_all.append(text_features.mean(0).unsqueeze(0))
            old_gpt_all = torch.cat(old_gpt_all, dim=0)
            old_gpt_all = old_gpt_all / old_gpt_all.norm(dim=-1, keepdim=True)
            print("Total CuPL prompt classes used for ZS evaluation: ", k)
            if torch.is_tensor(mean_text_features):
                mean_text_features = torch.cat([mean_text_features.unsqueeze(0), old_gpt_all.unsqueeze(0)], dim=0).mean(
                    0)
                mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)
                text_features = mean_text_features
            else:
                text_features = old_gpt_all
                mean_text_features = old_gpt_all
        if use_attribute_prompts:
            attribute_prompt_all = 0
            print("Using attribute templates for text embeddings")
            print(f"Prompt ensembling (n={len(ctx_templates)})")
            for i, temp in enumerate(ctx_templates):
                prompts = [temp.format(c.replace("_", " ")) for c in classnames]
                prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
                with torch.no_grad():
                    text_features = clip_model.encode_text(prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                attribute_prompt_all = attribute_prompt_all + text_features
            attribute_prompt_all = attribute_prompt_all / len(ctx_templates)
            attribute_prompt_all = attribute_prompt_all / attribute_prompt_all.norm(dim=-1, keepdim=True)
            if torch.is_tensor(mean_text_features):
                mean_text_features = torch.cat([mean_text_features.unsqueeze(0), attribute_prompt_all.unsqueeze(0)],
                                               dim=0).mean(0)
                mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)
                text_features = mean_text_features
            else:
                text_features = attribute_prompt_all
                mean_text_features = attribute_prompt_all

        # If above is set to False, use single prompt dataset conditioned template
        if not torch.is_tensor(mean_text_features):
            print("Performing single 1 template zeroshot inference")
            with torch.no_grad():
                text_features = clip_model.encode_text(prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class ZeroshotCLIP2(ZeroshotCLIP):
    """Prompt ensembling."""

    # templates = IMAGENET_TEMPLATES
    templates = IMAGENET_TEMPLATES

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        for params in clip_model.parameters():
            params.requires_grad_(False)

        # add custom-made prompt
        if cfg.DATASET.NAME != "ImageNet":
            self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model
