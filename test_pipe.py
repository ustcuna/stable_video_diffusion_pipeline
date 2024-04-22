import torch
import argparse
import time
import torch
import datetime
import psutil
from datetime import datetime, date, timedelta

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from pipeline_stable_video_diffusion_ipex import StableVideoDiffusionPipelineIpex

device="cpu"

def get_host_memory():
    memory_allocated = round(psutil.Process().memory_info().rss / 1024**3, 3)
    print("cpu"," memory used total: ", memory_allocated, "GB")

#model_id = "/home/models/stable-video-diffusion-img2vid-xt-1-1/"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help="model path for stable-video-diffusion-img2vid-xt-1-1")
    parser.add_argument('--bf16', default=False, action='store_true', help="FP32 - Default")

    args = parser.parse_args()

    model_id = args.model_path
    #pipe = StableVideoDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, variant="fp32")
    pipe = StableVideoDiffusionPipelineIpex.from_pretrained(model_id, torch_dtype=torch.float32, variant="fp32")
    pipe.to("cpu")

    image = load_image("./bali.jpg")
    #image = load_image("./test_image.png")
    image = image.resize((1024, 576))
    
    get_host_memory()
    
    data_type = torch.bfloat16 if args.bf16 else torch.float32

    pipe.prepare_for_ipex(image, data_type, height=576, width=1024, num_frames=25, num_inference_steps=30)
    
    generator = torch.Generator(device).manual_seed(4)
    with torch.no_grad(), torch.cpu.amp.autocast(enabled=args.bf16, dtype=torch.bfloat16):
        for i in range(1):
            print('')
            t1 = time.time()
            frames = pipe(image, num_frames=25, num_inference_steps=30, decode_chunk_size=8, generator=generator).frames[0]
            t2 = time.time()
            print('svd_xt_1_1 inference latency: {:.3f} sec'.format(t2-t1))
            print('******************************')
            print('')

        post_fix = "bf16" if args.bf16 else "fp32"
        export_to_video(frames, "generated_" + post_fix + ".mp4", fps=10)


