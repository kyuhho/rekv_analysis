import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def visualize_retrieval_attention(model, save_path="retrieval_attention.png", title="Retrieval Attention Heatmap"):
    """
    Visualizes the retrieval similarity scores across all layers and blocks.
    Y-axis: Layers
    X-axis: Blocks (Frames)
    Values: Similarity scores for retrieved blocks, 0 for others.
    """
    if not hasattr(model, 'last_retrieval_info') or not model.last_retrieval_info:
        print("No retrieval information found in model. Make sure you ran question_answering.")
        return

    # Matrix size: [Layers, Total number of blocks]
    num_layers = len(model.last_retrieval_info)
    
    # We need to find the total number of blocks
    # We can get this from the ContextManager in the model
    kv_cache = model.kv_cache
    if kv_cache is None:
        print("No KV cache found.")
        return
        
    sample_cm = kv_cache[0]
    total_blocks = sample_cm.num_global_block if hasattr(sample_cm, "num_global_block") else 0
    if total_blocks == 0:
        total_blocks = len(sample_cm.global_blocks[0]) if sample_cm.global_blocks and len(sample_cm.global_blocks) > 0 else 0

    if total_blocks == 0:
        # Fallback: find maximum index in retrieved indices
        max_idx = 0
        for info in model.last_retrieval_info:
            if info['indices']:
                max_idx = max(max_idx, max(info['indices'][0]))
        total_blocks = max_idx + 1

    attn_matrix = np.zeros((num_layers, total_blocks))

    for i, info in enumerate(model.last_retrieval_info):
        sims = info['similarity'] # [batch_size, num_blocks]
        indices = info['indices'] # [batch_size, topk]
        
        if sims is not None and indices is not None:
            batch_idx = 0
            sims_np = sims[batch_idx].numpy()
            indices_list = indices[batch_idx]
            
            # Fill selected ones, others stay 0
            for idx in indices_list:
                if idx < total_blocks:
                    attn_matrix[i, idx] = sims_np[idx]

    # 2. Plotting
    plt.figure(figsize=(20, 8))
    # Using a diverging or sequential colormap
    ax = sns.heatmap(attn_matrix, cmap="YlGnBu", cbar_kws={'label': 'Retrieval Similarity'})
    
    plt.xlabel("Video Block / Frame Index")
    plt.ylabel("Layer Index")
    plt.title(title)
    
    # Improve y-axis ticks
    plt.yticks(np.arange(num_layers) + 0.5, np.arange(num_layers))
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Visualization saved to {save_path}")

def wrap_and_visualize(vqa_instance, video_id, question, output_dir="results/visuals"):
    """
    Helper function to be called inside analyze_a_video
    """
    model = vqa_instance.qa_model
    # Clean up video_id and question for filename
    clean_video_id = str(video_id).split('/')[-1].replace('.', '_')
    clean_q = "".join([c if c.isalnum() else "_" for c in question[:30]])
    filename = f"{clean_video_id}_{clean_q}.png"
    save_path = os.path.join(output_dir, filename)
    visualize_retrieval_attention(model, save_path=save_path, title=f"Question: {question}")
