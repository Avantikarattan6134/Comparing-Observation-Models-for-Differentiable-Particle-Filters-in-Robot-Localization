import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results = {
    'CNN':          {'nav01': {'rmse': 85.41,  'ate': 47.89}, 'nav02': {'rmse': 141.61, 'ate': 79.80}, 'nav03': {'rmse': 240.57, 'ate': 154.07}},
    'ViT':          {'nav01': {'rmse': 83.65,  'ate': 45.66}, 'nav02': {'rmse': 155.28, 'ate': 94.89}, 'nav03': {'rmse': 308.98, 'ate': 223.06}},
    'ResNet':       {'nav01': {'rmse': 67.54,  'ate': 34.96}, 'nav02': {'rmse': 104.27, 'ate': 52.34}, 'nav03': {'rmse': 184.43, 'ate': 104.33}},
    'ResNet+LSTM':  {'nav01': {'rmse': 76.68,  'ate': 39.69}, 'nav02': {'rmse': 138.81, 'ate': 70.47}, 'nav03': {'rmse': 263.08, 'ate': 167.14}},
    'Swin':         {'nav01': {'rmse': 77.83,  'ate': 42.10}, 'nav02': {'rmse': 149.58, 'ate': 92.42}, 'nav03': {'rmse': 269.95, 'ate': 181.55}},
    'InEKF+ResNet': {'nav01': {'rmse': 326.47, 'ate': 301.69}, 'nav02': {'rmse': 497.89, 'ate': 472.77}, 'nav03': {'rmse': 595.14, 'ate': 571.99}},
}
models = list(results.keys())
tasks  = ['nav01', 'nav02', 'nav03']
colors = ['#2196F3','#FF9800','#4CAF50','#9C27B0','#00BCD4','#F44336']

def grouped_bar(metric, ylabel, filename):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(tasks)); w = 0.12
    for i,(m,c) in enumerate(zip(models,colors)):
        vals = [results[m][t][metric] for t in tasks]
        ax.bar(x+(i-2.5)*w, vals, w, label=m, color=c, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(['nav01 (Easy)','nav02 (Medium)','nav03 (Hard)'])
    ax.set_ylabel(ylabel); ax.legend(); ax.grid(axis='y', alpha=0.3)
    ax.set_title(f'{ylabel} - All Models Comparison')
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()
    print(f'Saved {filename}')

grouped_bar('rmse', 'RMSE (pixels)', 'plot_rmse_comparison.png')
grouped_bar('ate',  'ATE (pixels)',  'plot_ate_comparison.png')

# Difficulty scaling
fig, ax = plt.subplots(figsize=(10, 5))
for m,c in zip(models,colors):
    vals = [results[m][t]['rmse'] for t in tasks]
    ax.plot([1,2,3], vals, marker='o', label=m, color=c, linewidth=2)
ax.set_xticks([1,2,3]); ax.set_xticklabels(['nav01\n(Easy)','nav02\n(Medium)','nav03\n(Hard)'])
ax.set_ylabel('RMSE (pixels)'); ax.legend(); ax.grid(alpha=0.3)
ax.set_title('Performance vs Task Difficulty')
plt.tight_layout(); plt.savefig('plot_difficulty_scaling.png', dpi=150); plt.close()
print('Saved plot_difficulty_scaling.png')

# Particle vs Kalman
fig, ax = plt.subplots(figsize=(10, 5))
dpf4 = ['CNN','ViT','ResNet','ResNet+LSTM','Swin']
x = np.arange(len(tasks)); w = 0.25
ax.bar(x-w, [results['ResNet'][t]['rmse'] for t in tasks],   w, label='Best DPF (ResNet)',  color='#4CAF50', alpha=0.85)
ax.bar(x,   [np.mean([results[m][t]['rmse'] for m in dpf4]) for t in tasks], w, label='Avg DPF', color='#2196F3', alpha=0.85)
ax.bar(x+w, [results['InEKF+ResNet'][t]['rmse'] for t in tasks], w, label='InEKF+ResNet', color='#F44336', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(['nav01','nav02','nav03'])
ax.set_ylabel('RMSE (pixels)'); ax.legend(); ax.grid(axis='y', alpha=0.3)
ax.set_title('Particle Filter vs Kalman Filter')
plt.tight_layout(); plt.savefig('plot_particle_vs_kalman.png', dpi=150); plt.close()
print('Saved plot_particle_vs_kalman.png')
