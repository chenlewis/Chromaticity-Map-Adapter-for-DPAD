# Chromaticity-Map-Adapter-for-DPAD

## Overview

This is the implementation of the method proposed in "CMA: A Chromaticity Map Adapter for Robust Detection of Screen-Recapture Document Images" with pytorch(1.7.1, gpu version). The paper has been accepted by CVPR 2024. The codes are available soon.

## Introduction

In this work, we design the Chromaticity Map Adapter (CMA) network to extract chromaticity feature maps based on the pixel-level distortion model established in the paper. These feature maps are then fed into the Transformer network as multi-modal prompt tokens.
![Image text](https://github.com/chenlewis/Chromaticity-Map-Adapter-for-DPAD/blob/main/figures/Pixel.png)
