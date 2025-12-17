import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


def calc_average_metric(results, metric, task):
    task_results = [item for item in results if item['task'] == task]
    if len(task_results) == 0:
        return None
    metric_value = round(sum([item[metric] for item in task_results]) / len(task_results), 3)
    return metric_value

def plot_task_trends(final_results, save_dir, tasks):
    """각 task별로 실험들 간의 accuracy 변화 추이를 그래프로 저장"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 실험 이름들을 정렬 (32 → 48 → origin → 80 → 96 순서)
    def get_sort_key(exp_name):
        """실험 이름에서 정렬 키 추출"""
        # 예: "64-0.5-32-96" → 32, "64-0.5-origin" → 64
        if 'origin' in exp_name:
            return 64  # origin은 중간에 위치 (64)
        
        # "64-0.5-32-96" 형태에서 마지막 두 숫자 중 첫 번째 추출
        parts = exp_name.split('-')
        # 마지막 두 부분이 숫자인 경우 (예: "32-96")
        if len(parts) >= 4:
            try:
                # 마지막에서 두 번째 부분이 숫자면 그것을 사용 (32-96에서 32)
                if parts[-2].isdigit():
                    return int(parts[-2])
                # 아니면 마지막 부분 사용
                elif parts[-1].isdigit():
                    return int(parts[-1])
            except:
                pass
        return 999  # 숫자를 찾을 수 없으면 맨 뒤
    
    exp_names = sorted(final_results.keys(), key=get_sort_key)
    
    for task in tasks:
        # 각 실험의 해당 task accuracy 추출
        accuracies = []
        valid_exp_names = []  # 정렬/순서를 위한 원본 이름
        display_names = []     # 그래프 축에 표시할 축약 이름
        
        for exp_name in exp_names:
            if task in final_results[exp_name] and final_results[exp_name][task] is not None:
                accuracies.append(final_results[exp_name][task])
                valid_exp_names.append(exp_name)
                
                # 표시용 이름 생성
                if 'origin' in exp_name:
                    display_names.append('64-64')
                else:
                    parts = exp_name.split('-')
                    if len(parts) >= 2:
                        display_names.append(f'{parts[-2]}-{parts[-1]}')
                    else:
                        display_names.append(exp_name)
        
        if len(accuracies) == 0:
            continue
        
        # y축 범위 계산 (최솟값-15 ~ 최댓값+15)
        min_acc = min(accuracies)
        max_acc = max(accuracies)
        y_min = max(0, min_acc - 5)  # 0 이하로 내려가지 않도록
        y_max = min(100, max_acc + 5)  # 100 이상으로 올라가지 않도록
        
        # 그래프 생성
        plt.figure(figsize=(10, 6))
        plt.plot(valid_exp_names, accuracies, marker='o', linewidth=2.5, markersize=10)
        plt.xlabel('Experiment', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.title(f'{task}', fontsize=18, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(ticks=range(len(display_names)), labels=display_names, rotation=45, ha='right', fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim(y_min, y_max)
        
        # 각 점에 값 표시
        for i, (exp_name, acc) in enumerate(zip(valid_exp_names, accuracies)):
            plt.text(i, acc, f'{acc:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{task}_trend.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'Saved: {save_path}')

result_paths = [
    "/home/work/Redteaming/kyuho/strm/rekv_analysis/results/llava_ov_7b/mlvu/64-0.5-origin/results.csv",
    "/home/work/Redteaming/kyuho/strm/rekv_analysis/results/llava_ov_7b/mlvu/64-0.5-96-32/results.csv",
    "/home/work/Redteaming/kyuho/strm/rekv_analysis/results/llava_ov_7b/mlvu/64-0.5-80-48/results.csv",
    "/home/work/Redteaming/kyuho/strm/rekv_analysis/results/llava_ov_7b/mlvu/64-0.5-48-80/results.csv",
    "/home/work/Redteaming/kyuho/strm/rekv_analysis/results/llava_ov_7b/mlvu/64-0.5-32-96/results.csv",
]

save_dir = "/home/work/Redteaming/kyuho/strm/rekv_analysis/results/exp"

final_results = {}

for result_path in result_paths:
    df = pd.read_csv(result_path)
    results = df.to_dict(orient='records')

    exp_name = os.path.basename(os.path.dirname(result_path))  # "64-0.5-origin"

    if exp_name not in final_results:
        final_results[exp_name] = {}
    
    for task in ['plotQA', 'findNeedle', 'ego', 'count', 'order']:
        metric_value = calc_average_metric(results, 'qa_acc', task)
        final_results[exp_name][task] = metric_value

print(final_results)

# 각 task별 accuracy 변화 추이 그래프 생성
tasks = ['plotQA', 'findNeedle', 'ego', 'count', 'order']
plot_task_trends(final_results, save_dir, tasks)