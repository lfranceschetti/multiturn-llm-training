import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data
df = pd.read_csv('evaluations/evaluation_results_multi.csv')

# Llama baseline
llama = df[df['method']=='Llama']
llama_success = llama['percentage_successful_episodes'].iloc[0] * 100  # Convert to percentage

# Keep only rows with a step number
df = df.dropna(subset=['number'])
df['steps'] = df['number']

# Mapping for labels and colours
label_names = {
    'grpo_turn_level_multi_game_2_full': 'GRPO',
    'grpo_turn_level_multi_game_2': 'LA-GRPO',
}
colors = {
    'grpo_turn_level_multi_game_2_full': 'tab:orange',
    'grpo_turn_level_multi_game_2': 'tab:green',
}

# Create figure
plt.figure(figsize=(10, 8), dpi=100)

for method in df['method'].unique():
    if method not in label_names:
        continue

    mdata = df[df['method']==method]
    # Plot vs. training steps
    steps = [0] + mdata['steps'].tolist()
    success = [llama_success] + (mdata['percentage_successful_episodes'] * 100).tolist()  # Convert to percentage
    steps = np.array(steps)
    success = np.array(success)

    plt.plot(steps, success,
             marker='o',
             label=label_names[method],
             linewidth=3,
             markersize=8,
             color=colors[method],
             alpha=0.9)

# Axis labels
plt.xlabel('Training Steps', fontsize=14, fontweight='bold', labelpad=15)
plt.ylabel('Success Rate (%)', fontsize=14, fontweight='bold', labelpad=10)

# Grid, ticks, axis limits
plt.grid(True, linestyle='-', alpha=0.7, linewidth=1.0)
plt.tick_params(axis='x', pad=10, labelsize=12)
plt.tick_params(axis='y', pad=10, labelsize=12)
plt.xlim(0, 800)
plt.ylim(0, 100)

# Legend
plt.legend(frameon=True,
          facecolor='white',
          edgecolor='gray',
          fontsize=14,
          loc='upper center',
          ncol=len(label_names),
          bbox_to_anchor=(0.5, 1.02))

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('success_multi_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close() 