# Visual Prompt Tuning 

[https://arxiv.org/abs/2203.12119](https://cvpr.thecvf.com/virtual/2024/poster/30641) 


![Pixel](https://github.com/chenlewis/Chromaticity-Map-Adapter-for-DPAD/blob/main/figures/Pixel.png)

## Environment settings

See `env_setup.sh`

## Structure of the this repo (key files are marked with 👉):

- `src/configs`: handles config parameters for the experiments.
  
  * 👉 `src/config/config.py`: <u>main config setups for experiments and explanation for each of them. </u> 

- `src/data`: loading and setup input datasets. The `src/data/vtab_datasets` are borrowed from 

  [VTAB github repo](https://github.com/google-research/task_adaptation/tree/master/task_adaptation/data).


- `src/engine`: main training and eval actions here.

- `src/models`: handles backbone archs and heads for different fine-tuning protocols 

- 👉`train.py`: call this one for training and eval a model with a specified transfer type.
- 👉`tune_fgvc.py`: call this one for tuning learning rate and weight decay for a model with a specified transfer type. We used this script for FGVC tasks.
- 👉`tune_vtab.py`: call this one for tuning vtab tasks: use 800/200 split to find the best lr and wd, and use the best lr/wd for the final runs
- `launch.py`: contains functions used to launch the job.

## Experiments

### Key configs:

- 🔥CMA related:
  - MODEL.PROMPT.NUM_TOKENS: prompt length
  - MODEL.PROMPT.DEEP: deep or shallow prompt
- Fine-tuning method specification:
  - MODEL.TRANSFER_TYPE
- Vision backbones:
  - DATA.FEATURE: specify which representation to use
  - MODEL.TYPE: the general backbone type, e.g., "vit" or "swin"
  - MODEL.MODEL_ROOT: folder with pre-trained model checkpoints
- Optimization related: 
  - SOLVER.BASE_LR: learning rate for the experiment
  - SOLVER.WEIGHT_DECAY: weight decay value for the experiment
  - DATA.BATCH_SIZE
- Datasets related:
  - DATA.NAME
  - DATA.DATAPATH: where you put the datasets
  - DATA.NUMBER_CLASSES
- Others:
  - RUN_N_TIMES: ensure only run once in case for duplicated submision, not used during vtab runs
  - OUTPUT_DIR: output dir of the final model and logs
  - MODEL.SAVE_CKPT: if set to `True`, will save model ckpts and final output of both val and test set

### Datasets preperation:

-To fine-tune the model with your own data, you can refer to the data usage in the VPT project. For more details, visit the VPT GitHub repository: [VPT GitHub](https://github.com/KMnP/vpt).

-To preprocess your data, use the `docDatasetSplit.py` script located in the `dataprepare` folder. This script helps you split and prepare your data for subsequent model fine-tuning and training.

### Pre-trained model preperation

To prepare for training, you need to download and place the pre-trained Transformer-based backbones into the `MODEL.MODEL_ROOT` directory. 

1. **Download Pre-trained Models**: 
   - Download the backbone pre-trained models from the provided URL (VPT backbone models).

2. **Rename Pre-trained ViT Model**:
   - You also need to rename the downloaded ViT-B/16 checkpoint file from `ViT-B_16.npz` to `imagenet21k_ViT-B_16.npz`.

Ensure that all files are placed correctly and named appropriately for the model to load successfully.

### Training and Testing

For convenience, you can use the scripts in the `TrainOrTestCmdShell` folder for training and testing the model:

- **Training**: Use the `dataTrain.sh` script to train the model.
- **Testing**: Use the `CrossDatatest.sh` script to test the model.

These scripts simplify the process of running training and testing commands, allowing you to focus on the experimentation and evaluation of your models.

## Citation

If you find our work helpful in your research, please cite it as:

```
@inproceedings{jia2022vpt,
  title={Visual Prompt Tuning},
  author={Jia, Menglin and Tang, Luming and Chen, Bor-Chun and Cardie, Claire and Belongie, Serge and Hariharan, Bharath and Lim, Ser-Nam},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

## License

The majority of VPT is licensed under the CC-BY-NC 4.0 license (see [LICENSE](https://github.com/KMnP/vpt/blob/main/LICENSE) for details). Portions of the project are available under separate license terms: GitHub - [google-research/task_adaptation](https://github.com/google-research/task_adaptation) and [huggingface/transformers](https://github.com/huggingface/transformers) are licensed under the Apache 2.0 license; [Swin-Transformer](https://github.com/microsoft/Swin-Transformer), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) and [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) are licensed under the MIT license; and [MoCo-v3](https://github.com/facebookresearch/moco-v3) and [MAE](https://github.com/facebookresearch/mae) are licensed under the Attribution-NonCommercial 4.0 International license.
