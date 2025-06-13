import pandas as pd
import matplotlib.pyplot as plt
import re
import json
import seaborn as sns
import numpy as np

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')
# Use a more saturated color palette

# Read the CSV file
df = pd.read_csv('evaluations/evaluation_results.csv')

# Read the token counts from JSON
with open('evaluations/training_tokens_used.json', 'r') as f:
    token_counts = json.load(f)

# Get the Llama baseline values
llama_baseline_mean = df[df['method'] == 'Llama']['mean_reward'].iloc[0]
llama_baseline_std = df[df['method'] == 'Llama']['std_reward'].iloc[0]

# Function to extract steps from folder name
def extract_steps(folder):
    match = re.search(r'-(\d+)$', folder)
    if match:
        return int(match.group(1))
    return None

# Create a new column for steps
df['steps'] = df['number']

# Filter out rows without steps (like Llama-3.1-8B-Instruct)
df = df.dropna(subset=['number'])

# Create the plot with a larger figure size and custom DPI
plt.figure(figsize=(12, 8), dpi=100)

label_names = {
    'REFUEL-onesided-lora-beta-0.1-3': 'REFUEL',
    'grpo_onesided_1_starter_change': 'GRPO',
    'grpo_turn_level_onesided_2_starter_change': 'Multi-turn GRPO',
    'DPO_5': 'DPO',
}

# Define colors for each method
colors = {
    'REFUEL-onesided-lora-beta-0.1-3': 'tab:blue',
    'grpo_onesided_1_starter_change': 'tab:orange',
    'grpo_turn_level_onesided_2_starter_change': 'tab:green',
    'DPO_5': 'tab:red',
}

# Plot each method with different colors and enhanced styling
for method in df['method'].unique():
    method_data = df[df['method'] == method]
    method_key = method
    if method_key in token_counts and method in label_names:
        tokens = [0]
        rewards = [llama_baseline_mean]
        stds = [llama_baseline_std]
        
        for step, reward, std in zip(method_data['number'], method_data['mean_reward'], method_data['std_reward']):
            step_str = str(int(step))
            if step_str in token_counts[method_key]:
                tokens.append(token_counts[method_key][step_str])
                rewards.append(reward)
                stds.append(std)
        
        # Convert to numpy arrays for easier manipulation
        tokens_arr = np.array(tokens)
        rewards_arr = np.array(rewards)
        stds_arr = np.array(stds)
        
        # Plot the mean reward
        plt.plot(tokens_arr, rewards_arr, 
                marker='o', 
                label=label_names[method], 
                linewidth=3,
                markersize=8,
                color=colors[method],
                alpha=0.9)
        
        # Add error bands
        plt.fill_between(tokens_arr, 
                        rewards_arr - stds_arr, 
                        rewards_arr + stds_arr, 
                        alpha=0.2,
                        color=colors[method])

# Customize the plot appearance
plt.xlabel('Number of Training Tokens', fontsize=14, fontweight='bold', labelpad=15)
plt.ylabel('$\mathbf{R}(\mathbf{s}_{\mathbf{N+1}})$', fontsize=14, fontweight='bold', labelpad=15)

# Customize the legend
plt.legend(frameon=True, 
          facecolor='white', 
          edgecolor='gray',
          fontsize=12,
          loc='lower right')

# Customize the grid and ticks
plt.grid(True, linestyle='-', alpha=0.7, linewidth=1.0)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, 100)

# Add more padding between axis labels and tick labels
plt.gca().xaxis.set_tick_params(pad=10)
plt.gca().yaxis.set_tick_params(pad=10)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot with high quality
plt.savefig('ra_tokens_reward_std.png', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')
plt.close()