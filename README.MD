# DiffCom

> Unified Geometry and Color Compression Framework for Point Clouds via Generative Diffusion Priors  
> 3DV 2026 • [Project Page](https://tianxinhuang.github.io/)

## Overview
DiffCom injects generative diffusion priors into point-cloud compression, jointly optimizing geometry and color. It supports textured/textureless assets, enabling bitrate–distortion trade-offs across compression levels 1–9.

## Prerequisites
- **OS**: Ubuntu 20.04/22.04 (tested), macOS for orchestration only  
- **GPU**: ≥16 GB VRAM (A4000 or better recommended)  
- **CUDA**: 12.4 (match with Torch 2.6.0 wheels)  
- **Python**: 3.10.13 (Conda environment suggested)

## Installation
1. **Create environment**
   ```bash
   conda create -n diffcom python=3.10.13
   conda activate diffcom
   ```
2. **Install CUDA 12.4 toolchain**
   ```bash
   conda install -c "nvidia/label/cuda-12.4.0" cuda
   ```
3. **Build MPEG-GPCC (tmc3)**
   ```bash
   cd Diffcom/point_e
   mkdir build && cd build
   cmake .. && make -j"$(nproc)"
   cd ../..
   ```
4. **Install PyTorch (cu124)**
   ```bash
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
     --index-url https://download.pytorch.org/whl/cu124
   ```
5. **Install Point-E environment**
   ```bash
   pip install -e .
   ```
6. **Compile PointNet++ ops**
   ```bash
   cd pointnet2_ops_lib
   python setup.py install
   cd ..
   ```

## Datasets & Checkpoints
- **Evaluation data**: [HuggingFace – DiffCom_Data](https://huggingface.co/datasets/huangtx/Diffcom_Data/tree/main)  
- **Pretrained models**: Automatically cached under `point_e_model_cache/`.  
  - To sidestep throttled downloads, place checkpoints manually in this folder with matching filenames.

## Quick Start
Navigate to `Diffcom/point_e` and prepare:
- `--indir`: directory of input point clouds (`.ply`, `.pcd`, etc.)
- `--drcdir`: directory for compressed `.bin` files
- `--dedir`: directory for decoded outputs
- `--workspace`: scratch directory for intermediate states

### Colored point clouds
```bash
python compress.py \
  --level-list 5 (1 to 7) \
  --drcdir path/to/bin \
  --dedir path/to/output \
  --indir path/to/input \
  --iterations 1500 \
  --base-name base40M \
  --query-radius 0.05 \
  --query-max-neighbors 31 \
  --workspace path/to/workspace
```

### Geometry-only point clouds
```bash
python compress.py \
  --level-list 5 (1 to 7) \
  --drcdir path/to/bin \
  --dedir path/to/output \
  --indir path/to/input \
  --iterations 1500 \
  --base-name base40M \
  --query-radius 0.01 \
  --query-max-neighbors 31 \
  --workspace path/to/workspace
```

- Compressed binaries → `your_bin_dir`  
- Reconstructed point clouds → `your_output_dir`  
- Adjust `--level-list`, `--query-radius`, `--query-max-neighbors`, and `--iterations` according to density and target bitrate.

## Citation
```bibtex
@article{huang2025unified,
  title={Unified Geometry and Color Compression Framework for Point Clouds via Generative Diffusion Priors},
  author={Huang, Tianxin and Lee, Gim Hee},
  journal={arXiv preprint arXiv:2503.18083},
  year={2025}
}
```

## Acknowledgements & Contact
- Built upon [Point-E](https://openai.com/index/point-e/). Grateful for the original release.
- Contact: 21725129@zju.edu.cn