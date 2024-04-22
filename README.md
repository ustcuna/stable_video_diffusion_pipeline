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
conda install jemalloc==5.2.1
```
Install dependencyies
```
pip install -r requirements.txt
```
## Run optimized SVD solution using diffusers pipeline
```
# Pls change your own ENV_PATH and model_path in the script
sh run_pipe.sh
```
