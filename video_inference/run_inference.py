import os
import argparse
import subprocess
import multiprocessing

def run_inference(args):
    cmd = ["python", f"video_inference/rekv_stream_inference.py",
           "--model", args.model,
           "--video_path", args.video_path,
           "--sample_fps", str(args.sample_fps),
           "--n_local", str(args.n_local),
           "--retrieve_size", str(args.retrieve_size),
           "--debug", args.debug]
    
    print(f'exec: {cmd}')
    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava_ov_7b", choices=['llava_ov_0.5b', 'llava_ov_7b'])
    parser.add_argument("--video_path", type=str, default="/data/seohyeong2/streaming/tt0067116.npy")
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--n_local", type=int, default=15000)
    parser.add_argument("--retrieve_size", type=int, default=64)
    parser.add_argument("--debug", type=str, default='false')
    args = parser.parse_args()
 
    run_inference(args)
