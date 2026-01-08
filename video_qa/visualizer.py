import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import hashlib

def visualize_retrieval_attention(model, save_path="retrieval_attention.png", title="Retrieval Attention Heatmap"):
    """
    Visualizes the ACTUAL attention scores used during inference.
    Y-axis: Layers
    X-axis: Blocks (Frames)
    Values: Softmax attention scores sum per block.
    """
    if not hasattr(model, 'last_retrieval_info') or not model.last_retrieval_info:
        return

    num_layers = len(model.last_retrieval_info)
    kv_cache = model.kv_cache
    if kv_cache is None: return
        
    sample_cm = kv_cache[0]
    total_blocks = sample_cm.num_global_block if hasattr(sample_cm, "num_global_block") else 0
    if total_blocks == 0:
        total_blocks = len(sample_cm.global_blocks[0]) if sample_cm.global_blocks and len(sample_cm.global_blocks) > 0 else 0

    if total_blocks == 0:
        max_idx = 0
        for info in model.last_retrieval_info:
            if info['indices']: max_idx = max(max_idx, max(info['indices'][0]))
        total_blocks = max_idx + 1

    attn_matrix = np.zeros((num_layers, total_blocks))
    n_init = sample_cm.n_init
    block_size = sample_cm.block_size

    for i, info in enumerate(model.last_retrieval_info):
        # attn_scores: (batch_size, num_heads, kv_len)
        scores = info['attn_scores']
        indices = info['indices']
        
        if scores is not None and indices is not None:
            batch_idx = 0
            # Sum across all heads and all query tokens (input_len)
            # info['attn_scores'] already comes from tmp.sum(dim=-2) in rekv_attention.py
            # which is (batch, heads, kv_len)
            
            # Sum across heads to get (kv_len,)
            token_scores = scores[batch_idx].sum(dim=0).numpy()
            
            # The structure of keys in init_h_k is [init_keys (n_init), retrieved_keys (topk * block_size)]
            retrieved_scores = token_scores[n_init:]
            indices_list = indices[batch_idx]
            
            # Map token-level scores back to blocks
            for b_cnt, real_block_idx in enumerate(indices_list):
                if real_block_idx < total_blocks:
                    st = b_cnt * block_size
                    ed = st + block_size
                    if ed <= len(retrieved_scores):
                        # Sum attention scores within the block
                        attn_matrix[i, real_block_idx] = retrieved_scores[st:ed].sum()

    # 2. Plotting
    plt.figure(figsize=(24, 8))
    mask = (attn_matrix == 0)
    
    # Use a per-layer normalization for better visibility if needed, 
    # but here we show absolute attention weights across layers.
    ax = sns.heatmap(attn_matrix, 
                     mask=mask, 
                     cmap="YlGnBu", 
                     cbar_kws={'label': 'Total Attention Weight'},
                     linewidths=0, 
                     linecolor='white')
    
    ax.set_facecolor('white')
    plt.xlabel("Video Block / Frame Index")
    plt.ylabel("Layer Index (0=Bottom, Top=Last)")
    plt.title(title, fontsize=12)
    
    plt.yticks(np.arange(num_layers) + 0.5, np.arange(num_layers))
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Visualization saved to {save_path}")

def wrap_and_visualize(vqa_instance, video_id, question, q_idx=0, output_dir="results/visuals"):
    model = vqa_instance.qa_model
    clean_video_id = str(video_id).split('/')[-1].replace('.', '_')
    short_q = "".join([c if c.isalnum() else "_" for c in question[:30]])
    q_hash = hashlib.md5(question.encode()).hexdigest()[:6]
    
    filename = f"{clean_video_id}_Q{q_idx:02d}_{short_q}_{q_hash}.png"
    save_path = os.path.join(output_dir, filename)
    visualize_retrieval_attention(model, save_path=save_path, title=f"Question: {question}")
