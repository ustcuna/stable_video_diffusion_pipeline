ENV_PATH=/home/conda/envs/svd_xt/

export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0
# export DNNL_VERBOSE=1

export LD_PRELOAD=$ENV_PATH/lib/libiomp5.so
export LD_PRELOAD=$ENV_PATH/lib/libjemalloc.so:$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

numactl --localalloc --physcpubind=0-47 python test_pipe.py --model_path "/home/models/stable-video-diffusion-img2vid-xt-1-1/" --bf16
