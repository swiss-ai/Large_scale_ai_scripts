# large_scale_ai_scripts

### Setup
- Megatron codebase
    -   `git clone --branch core_v0.16.0 https://github.com/NVIDIA/Megatron-LM.git`
- Image
    - Use the `.toml` file provided
- Slurm Script
    - A script configured with Qwen2.5-14B dense model is provided and you should be able to configure your desired model size.
    - You have to configure your `MEGATRON_LM_DIR`, `DATASET_CACHE_DIR`, `IMAGE_ENV` and other variables manually.

### Usage
``` sh
cd /path/to/your/project
sbatch qwen2_5_dense.sh
```