import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

# Read data
df = pd.read_csv('evaluations/evaluation_results_multi.csv')
with open('evaluations/training_tokens_used_multi.json', 'r') as f:
    token_counts = json.load(f)

# Llama baseline
llama = df[df['method']=='Llama']
llama_mean = llama['mean_reward'].iloc[0]
llama_std  = llama['std_reward'].iloc[0]

# Keep only rows with a step number
df = df.dropna(subset=['number'])
df['steps'] = df['number']

# Mapping for labels and colours
label_names = {
    'grpo_turn_level_multi_game_2_full': 'GRPO',
    'grpo_turn_level_multi_game_2': 'LA-GRPO',  # renamed here
}
colors = {
    'grpo_turn_level_multi_game_2_full': 'tab:orange',
    'grpo_turn_level_multi_game_2': 'tab:green',
}

# Create side-by-side axes
fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=100)

for method in df['method'].unique():
    if method not in label_names:
        continue

    mdata = df[df['method']==method]
    # --- Plot vs. training steps ---
    steps   = [0] + mdata['steps'].tolist()
    rewards = [llama_mean] + mdata['mean_reward'].tolist()
    stds    = [llama_std]  + mdata['std_reward'].tolist()
    steps = np.array(steps)
    rewards = np.array(rewards)
    stds    = np.array(stds)

    axes[0].plot(steps, rewards,
                 marker='o',
                 label=label_names[method],
                 linewidth=3,
                 markersize=8,
                 color=colors[method],
                 alpha=0.9)
    axes[0].fill_between(steps,
                         rewards - stds,
                         rewards + stds,
                         alpha=0.2,
                         color=colors[method])

    # --- Plot vs. training tokens ---
    tokens   = [0]
    t_rewards = [llama_mean]
    t_stds    = [llama_std]
    for st, rv, sd in zip(mdata['steps'], mdata['mean_reward'], mdata['std_reward']):
        key = str(int(st))
        if key in token_counts[method]:
            tokens.append(token_counts[method][key])
            t_rewards.append(rv)
            t_stds.append(sd)
    tokens    = np.array(tokens)
    t_rewards = np.array(t_rewards)
    t_stds    = np.array(t_stds)

    axes[1].plot(tokens, t_rewards,
                 marker='o',
                 linewidth=3,
                 markersize=8,
                 color=colors[method],
                 alpha=0.9)
    axes[1].fill_between(tokens,
                         t_rewards - t_stds,
                         t_rewards + t_stds,
                         alpha=0.2,
                         color=colors[method])

# Axis labels (y-label pad reduced to bring it closer)
axes[0].set_xlabel('Training Steps', fontsize=14, fontweight='bold', labelpad=15)
axes[1].set_xlabel('Number of Training Tokens', fontsize=14, fontweight='bold', labelpad=15)
axes[0].set_ylabel(r'$\mathbf{R}(\mathbf{s}_{\mathbf{N+1}})$', fontsize=16, fontweight='bold', labelpad=10)
axes[1].set_ylabel(r'$\mathbf{R}(\mathbf{s}_{\mathbf{N+1}})$', fontsize=16, fontweight='bold', labelpad=10)

# Grid, ticks, axis limits
for ax in axes:
    ax.grid(True, linestyle='-', alpha=0.7, linewidth=1.0)
    ax.tick_params(axis='x', pad=10, labelsize=12)
    ax.tick_params(axis='y', pad=10, labelsize=12)
axes[0].set_xlim(0, 800)
axes[1].set_ylim(0, 100)
axes[0].set_ylim(0, 100)

# Shared legend on top, larger font
fig.legend(frameon=True,
           facecolor='white',
           edgecolor='gray',
           fontsize=14,
           loc='upper center',
           ncol=len(label_names),
           bbox_to_anchor=(0.5, 1.02))

# Adjust layout to leave room for the legend
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('combined_multi_plots.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
