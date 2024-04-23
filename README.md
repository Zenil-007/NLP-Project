<p align="center">
  <h1 align="center">Bootstrapping Large Language Models for Radiology Report Generation</h1>

The official GitHub repository of the AAAI-2024 paper ["Bootstrapping Large Language Models for Radiology Report Generation"](https://ojs.aaai.org/index.php/AAAI/article/view/29826).

# Reference
If our work is helpful to your research, please cite our paper:
``` latex
@inproceedings{chang2024bootstrapping,
  author       = {Chang Liu and
                  Yuanhe Tian and
                  Weidong Chen and
                  Yan Song and
                  Yongdong Zhang},
  editor       = {Michael J. Wooldridge and
                  Jennifer G. Dy and
                  Sriraam Natarajan},
  title        = {Bootstrapping Large Language Models for Radiology Report Generation},
  booktitle    = {AAAI},
  pages        = {18635--18643},
  year         = {2024},
}
```

# Getting Started
1. Before you run the code, you need to create a virtual environment and activate it via the following command:
```bash
conda env create -f environment.yaml
conda activate venv
```

2. Once the virtual environment is created, you need to download the LLM model weights following the instruction in [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4). Once the model weights are downloaded, you need to modify some configuration files:
- `minigpt4/models/minigpt4-7b.yaml`: line 16 with the path of Vicuna 7b model weights.
- `minigpt4/models/minigpt4.yaml`: line 16 with the path of Vicuna 13b model weights.

3. You need to download the dataset from the official websites of [IU X-Ray](https://openi.nlm.nih.gov/faq#collection) and [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/). Once the datasets are ready, you need to modify some configuration files:
- `minigpt4/configs/datasets/iuxray/align.yaml`: line 5 with the path of pre-training dataset.
- `minigpt4/configs/datasets/iuxray/generate_then_refine.yaml`: line 5 with the path of IU X-Ray dataset, line 6 with the path of public medical corpora.
- `minigpt4/configs/datasets/mimic/align.yaml`: line 5 with the path of pre-training dataset.
- `minigpt4/configs/datasets/mimic/generate_then_refine.yaml`: line 5 with the path of MIMIC-CXR dataset, line 6 with the path of public medical corpora.

# Training
1. **Pre-training.** We recommend you to follow the instructions below to pre-train MiniGPT-4 on MIMIC-CXR.

(1) Modify the configuration files.
- `train_configs/stage1/config.yaml`: line 12 with the path of the linear projection layer of MiniGPT-4, line 59 with the output path.

(2) Run the following command lines to pre-train MiniGPT-4 on MIMIC-CXR.
```
python train.py --cfg-path train_configs/stage1/config.yaml
```

If you need to reduce the memory usage, we recommend you to use the first stage strategy of `ZeRO` optimizer. Run the following command lines to pre-train MiniGPT-4 on MIMIC-CXR with a lower memory usage.

```
deepspeed --nproc-per-gpu NUM_GPUS --master-port MASTER_PORT train.py --cfg-path train_configs/stage1/config.yaml use_zero_optimizer --deepspeed_config train_configs/stage1/zero.json
```

You can download our pre-trained model weights from [here](https://huggingface.co/a-b-c-d-e-g/R2-LLM).

2. **Fine-tuning.** We recommend you to follow the instructions below to fine-tune MiniGPT-4 on IU X-Ray and MIMIC-CXR.

(1) Modify the configuration files. Herein, we take the IU X-Ray configuration as an example.
- `train_configs/stage2/iuxray/config.yaml`: line 11 with the path of the linear projection layer of pre-trained MiniGPT-4 on MIMIC-CXR, line 56 with the output path.

(2) Run the following command lines to fine-tune MiniGPT-4.

```
python train.py --cfg-path train_configs/stage2/iuxray/config.yaml
```

Our codebase supports `ZeRO` to reduce the memory usage. You can run the following command lines with `ZeRO`.

```
deepspeed --nproc-per-gpu NUM_GPUS --master-port MASTER_PORT train.py --cfg-path train_configs/stage2/iuxray/config.yaml use_zero_optimizer --deepspeed_config train_configs/stage2/iuxray/zero.json
```

You can download our fine-tuned model weights from [here](https://huggingface.co/a-b-c-d-e-g/R2-LLM).

# Inference
Run the following command lines to generate radiology reports.

```
python generate_reports.py \
--cfg-path configs/eval_configs/eval.yaml \
--gpu-id GPU_IDS \
--image_path IMAGE_PATH \
--annotations ANNOTATIONS_PATH_OF_IUXRAY_OR_MIMIC \
--checkpoint PATH_TO_PRETRAINED_MODEL_WEIGHTS \
```

# Acknowledgement
This GitHub repository is heavily built based on the [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) repository. Thanks to the authors for their great work!