import os
import argparse
import subprocess
import multiprocessing

# def eval_cgbench(args):
#     num_chunks = args.num_chunks
#     save_dir = f"results/{args.model}/cgbench/{args.retrieve_size}-{args.sample_fps}"
#     solver = "rekv_offline_vqa"
#     if not args.only_eval:
#         # QA
#         processes = []
#         for idx in range(0, num_chunks):
#             cmd = ["python", f"video_qa/{solver}.py",
#                     "--model", args.model,
#                     "--sample_fps", str(args.sample_fps),
#                     "--n_local", str(args.n_local),
#                     "--retrieve_size", str(args.retrieve_size),
#                     "--save_dir", save_dir,
#                     "--anno_path", "data/cgbench/full_mc.json",
#                     "--debug", args.debug,
#                     "--num_chunks", str(num_chunks),
#                     "--chunk_idx", str(idx)]
#             p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
#             processes.append(p)
#             p.start()
#         for p in processes:
#             p.join()
#         # merge results
#         exec(f"> {save_dir}/results.csv")
#         for idx in range(num_chunks):
#             if idx == 0:
#                 exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
#             exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
#             exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
#     # eval
#     exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")


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
    parser.add_argument("--model", type=str, default="llava_ov_7b", choices=['llava_ov_0.5b', 'llava_ov_7b', 'llava_ov_72b', 'video_llava_7b', 'longva_7b'])
    parser.add_argument("--video_path", type=str, default="/data/seohyeong2/streaming/tt0067116.npy")
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--n_local", type=int, default=15000)
    parser.add_argument("--retrieve_size", type=int, default=64)
    parser.add_argument("--debug", type=str, default='false')
    args = parser.parse_args()
 
    run_inference(args)
