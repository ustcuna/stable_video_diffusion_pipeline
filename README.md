# stable_video_diffusion_pipeline
Provide an optimized pipeline for stabilityai/stable-video-diffusion-img2vid-xt-1-1 on CPU.
## Optimizations
* Using AMX-BF16(Advanced Matrix Extensions) on Xeon SPR(4th gen)/EMR(5th gen) CPU
* Enable IPEX(Intel-Extension-for-Pytorch) optimization
* JIT trace for pipeline modules into TorchScript
* Optimized Softmax implementation especially for long sequences attention
* Enable jemalloc and intel-openmp

## Set up environment
Create conda env
```
conda create -n svd_xt python=3.10
conda activate svd_xt
conda install jemalloc
```
Install dependencyies
```
pip install torch==2.1.0 transformers diffusers accelerate mkl intel-openmp
pip install intel_extension_for_pytorch-2.1.0+git383aedd-cp310-cp310-linux_x86_64.whl
```
## Run optimized SVD solution using diffusers pipeline
```
# Pls change your own ENV_PATH and model_path in the script
sh run_pipe.sh
```
