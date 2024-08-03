# Visual Prompt Tuning 

[https://arxiv.org/abs/2203.12119](https://cvpr.thecvf.com/virtual/2024/poster/30641) 


![Pixel](https://github.com/chenlewis/Chromaticity-Map-Adapter-for-DPAD/blob/main/figures/Pixel.png)

## Environment settings

Please refer to the `env_setup.sh` script for setting up the environment. This script is borrowed from the VPT project and can be found here: [VPT GitHub](https://github.com/KMnP/vpt).

The `env_setup.sh` script will help you configure the necessary dependencies and environment settings required to run the project smoothly.

## Experiments

![CMA](https://github.com/chenlewis/Chromaticity-Map-Adapter-for-DPAD/blob/main/figures/CMA.png)

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
@inproceedings{chen2024cma,
  title={CMA: A Chromaticity Map Adapter for Robust Detection of Screen-Recapture Document Images},
  author={Chen, Changsheng and Lin, Liangwei and Chen, Yongqi and Li, Bin and Zeng, Jishen and Huang, Jiwu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15577--15586},
  year={2024}
}
```

## Acknowledgements

We would like to express our sincere gratitude to the authors of the VPT project for their invaluable contributions. The open-source code provided by the VPT team has been instrumental in the development of this project. 

Special thanks to the VPT team for their dedication to advancing research and providing a robust foundation for others to build upon. You can find their work here: [VPT GitHub](https://github.com/KMnP/vpt).



