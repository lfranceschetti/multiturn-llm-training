import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

# Read data
df = pd.read_csv('evaluations/evaluation_results.csv')
with open('evaluations/training_tokens_used.json', 'r') as f:
    token_counts = json.load(f)

# Baseline success rate at 0 steps
baseline_success = 87.2  # Using the specified baseline value

# Keep only rows with a step number
df = df.dropna(subset=['number'])
df['steps'] = df['number']

# Mapping for labels and colours
label_names = {
    'REFUEL-onesided-lora-beta-0.1-3': 'REFUEL',
    'grpo_onesided_1_starter_change': 'GRPO',
    'grpo_turn_level_onesided_2_starter_change': 'LA-GRPO',
    'DPO_5': 'DPO',
}
colors = {
    'REFUEL-onesided-lora-beta-0.1-3': 'tab:blue',
    'grpo_onesided_1_starter_change': 'tab:orange',
    'grpo_turn_level_onesided_2_starter_change': 'tab:green',
    'DPO_5': 'tab:red',
}

# Create figure
plt.figure(figsize=(10, 8), dpi=100)

for method in df['method'].unique():
    if method not in label_names:
        continue

    mdata = df[df['method']==method]
    # Plot vs. training tokens
    tokens = [0]
    success = [baseline_success]
    for st, succ in zip(mdata['steps'], mdata['percentage_successful_episodes']):
        key = str(int(st))
        if key in token_counts[method]:
            tokens.append(token_counts[method][key])
            success.append(succ * 100)  # Convert to percentage
    tokens = np.array(tokens)
    success = np.array(success)

    plt.plot(tokens, success,
             marker='o',
             label=label_names[method],
             linewidth=3,
             markersize=8,
             color=colors[method],
             alpha=0.9)

# Axis labels
plt.xlabel('Number of Training Tokens', fontsize=14, fontweight='bold', labelpad=15)
plt.ylabel('Success Rate (%)', fontsize=14, fontweight='bold', labelpad=10)

# Grid, ticks, axis limits
plt.grid(True, linestyle='-', alpha=0.7, linewidth=1.0)
plt.tick_params(axis='x', pad=10, labelsize=12)
plt.tick_params(axis='y', pad=10, labelsize=12)
plt.ylim(60, 100)

# Legend
plt.legend(frameon=True,
          facecolor='white',
          edgecolor='gray',
          fontsize=14,
          loc='lower center',
          ncol=len(label_names),
          bbox_to_anchor=(0.5, 1.05))

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('success_ra_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close() 