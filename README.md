# UDCNet

**Two-stage Dual-network Approach for Multi-depth Computer-generated Holography**


[![Paper](https://img.shields.io/badge/Paper-Optics%20%26%20Lasers%20in%20Engineering-blue)]()
[![GitHub Stars](https://img.shields.io/github/stars/Ryancheng236/UDCNet?style=social)](https://github.com/Ryancheng236/UDCNet)

---

## Highlights

- A two-stage dual-network model (UDCNet) is proposed for multi-depth computer-generated holography.
- The model outputs high-quality phase holograms for multiple specified depths in a single forward propagation.
- Random depth parameters and a combined loss function with learnable parameters are introduced during training.
- Average PSNR per depth layer exceeds 30 dB, with average SSIM above 0.9 at 20–30 cm depth.
- Proposed method outperforms SGD, DPAC and U-Net in efficiency and reconstruction quality.

---

## Method Overview

```
<img width="6330" height="2256" alt="UDCNet2" src="https://github.com/user-attachments/assets/cb87bb20-efd9-4baf-8fb6-68b25a891cc8" />

```

**UDCNet** consists of two stages:

1. **Stage I -- U-Net (Phase Encoder):** Maps the target amplitude image to a coarse phase encoding. The predicted phase is constrained to [-pi, pi] via Hardtanh activation.

2. **Stage II -- DCNet (Dual-Channel Refiner):** A dual-channel architecture that jointly processes amplitude and phase features:
   - **Upper channel** (U-Net-based): Extracts global structural information via DO-Conv and CRAB modules
   - **Lower channel** (ResNet-based): Captures local detail information via HDCRAB with hybrid dilated convolutions
   - **SDFB** (Semantic Difference Fusion Block): Fuses features from both channels to generate the final phase hologram

The angular spectrum method (ASM) is used for wave propagation between the SLM plane and target planes.

---

## Key Results

### Quantitative Comparison (at d = 20 cm)

| Method            | PSNR (dB) |   SSIM    | Inference Time |
| :---------------- | :-------: | :-------: | :------------: |
| **UDCNet (Ours)** | **30.35** | **0.935** | **118.66 ms**  |

<img width="3510" height="2083" alt="不同方法对比实验图-第 17 页" src="https://github.com/user-attachments/assets/7126f1ee-3820-488e-99f2-9e6cd373444e" />


### Multi-depth Reconstruction Quality

| Depth | Mean PSNR (dB) |    Mean SSIM    |
| :---: | :------------: | :-------------: |
| 20 cm | 31.36 +/- 1.68 | 0.947 +/- 0.013 |
| 25 cm | 30.80 +/- 1.58 | 0.938 +/- 0.017 |
| 30 cm | 30.42 +/- 1.99 | 0.931 +/- 0.023 |

<img width="613" height="645" alt="image" src="https://github.com/user-attachments/assets/5a4b6a9e-7925-4ca8-989e-5600844eb803" />

<img width="613" height="636" alt="image" src="https://github.com/user-attachments/assets/bc12debc-69d4-41f4-b666-f70b50292db8" />


---
<img width="1070" height="345" alt="image" src="https://github.com/user-attachments/assets/fb2479f5-86a4-419e-b455-64a17ffaad10" />
<img width="710" height="564" alt="image" src="https://github.com/user-attachments/assets/48abbd20-5471-41bc-87ce-b8c54f33cc79" />



## Requirements

- Python >= 3.11
- PyTorch >= 2.0.1
- NVIDIA GPU (tested on A100)


## Dataset

We use the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset for training:

- 800 images for training
- 100 images for validation
- All RGB images are converted to grayscale and resized to SLM resolution (1072 x 1920)

## Optical Configuration

| Parameter             | Value                                  |
| :-------------------- | :------------------------------------- |
| Wavelength            | 532 nm                                 |
| Pixel pitch           | 8 um                                   |
| Resolution            | 1072 x 1920                            |
| SLM                   | HoloEye Pluto (reflective, phase-only) |
| Reconstruction depths | 20 cm, 25 cm, 30 cm                    |


## Project Structure

```
UDCNet/
├── models/
│   ├── unet.py          # Stage I: U-Net phase encoder
│   ├── dcnet.py         # Stage II: Dual-channel refinement network
│   └── modules.py       # DO-Conv, CRAB, HDCRAB, SDFB modules
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{cheng2026two,
  title={Two-stage dual-network approach for multi-depth computer-generated holography},
  author={Cheng, Yazhou and Zhao, Fen and Liu, Chenhua and Dong, Mingli and Zhu, Lianqing},
  journal={Optics \& Laser Technology},
  volume={202},
  pages={115392},
  year={2026},
  publisher={Elsevier}
}
```

## Acknowledgements

This work is supported by the Research Project of Beijing Municipal Natural Science Foundation (No. BJXZ2021-012-00046).

## Contact

For questions or collaborations, please contact:

- Yazhou Cheng: [ryancheng236@github](https://github.com/Ryancheng236)

## License

This project is released for academic research purposes.
