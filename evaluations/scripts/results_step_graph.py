import pandas as pd
import matplotlib.pyplot as plt
import re
import json
import seaborn as sns

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')

# Read the CSV file
df = pd.read_csv('evaluations/evaluation_results.csv')

# Get the Llama baseline value
llama_baseline = df[df['method'] == 'Llama']['mean_reward'].iloc[0]

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

# Plot each method with different colors and enhanced styling
for method in df['method'].unique():
    method_data = df[df['method'] == method]
    
    # Add the baseline point (step 0 = Llama baseline)
    steps = [0]
    rewards = [llama_baseline]
    
    # Add the points for each step
    steps.extend(method_data['steps'].tolist())
    rewards.extend(method_data['mean_reward'].tolist())
    
    if method in label_names:
        plt.plot(steps, rewards, 
                marker='o', 
                label=label_names[method], 
                linewidth=3,
                markersize=10,
                alpha=0.9)

# Customize the plot appearance
plt.xlabel('Training Steps', fontsize=14, fontweight='bold', labelpad=15)
plt.ylabel('Reward $\mathbf{E}[\mathbf{R}(\mathbf{s}_{\mathbf{N+1}})]$', fontsize=14, fontweight='bold', labelpad=15)


# Customize the legend
plt.legend(frameon=True, 
          facecolor='white', 
          edgecolor='gray',
          fontsize=12,
          loc='lower right',
          bbox_to_anchor=(1.0, 0.0))

# Customize the grid and ticks
plt.grid(True, linestyle='-', alpha=1.0, linewidth=1.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0, 4000)

# Add more padding between axis labels and tick labels
plt.gca().xaxis.set_tick_params(pad=10)
plt.gca().yaxis.set_tick_params(pad=10)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot with high quality
plt.savefig('ra_steps_reward.png', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')
plt.close()