import torch
import argparse
import time
import torch
import os
import glob
import datetime
import psutil
from datetime import datetime, date, timedelta
from torchvision.transforms import ToTensor

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
    
    get_host_memory()
    
    data_type = torch.bfloat16 if args.bf16 else torch.float32

    images = []
    image_dir = "./input_images"
    allowed_extensions = [".jpg", ".png", ".jpeg"]  # Add more extensions as needed
    image_files = [file for file in glob.glob(os.path.join(image_dir, "*")) if os.path.splitext(file)[1] in allowed_extensions]

    for image_file in image_files:
        image = load_image(image_file)
        image = image.resize((1024, 576))
        images.append(image)
    
    generator = torch.Generator(device).manual_seed(4)
    with torch.no_grad(), torch.cpu.amp.autocast(enabled=args.bf16, dtype=torch.bfloat16):
        prepared = False
        index = 0
        for i in images:
            if not prepared:
                pipe.prepare_for_ipex(i, data_type, height=576, width=1024, num_frames=25, num_inference_steps=30)
                prepared = True
            t1 = time.time()
            frames = pipe(i, num_frames=25, num_inference_steps=30, decode_chunk_size=8, generator=generator).frames[0]
            t2 = time.time()
            print('svd_xt_1_1 inference latency: {:.3f} sec'.format(t2-t1))
            print('******************************')
            print('')
            post_fix = "bf16" if args.bf16 else "fp32"
            export_to_video(frames, "generated_" + str(index) + "_" + post_fix + ".mp4", fps=10)
            index += 1