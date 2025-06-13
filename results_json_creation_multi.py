import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import re

def extract_method_and_number(folder_name):
    # Special case for Llama
    if folder_name == "Llama-3.1-8B-Instruct":
        return ("Llama", 0)
    
    
    try:
        number = int(folder_name.split("-")[-1])
        return ("-".join(folder_name.split("-")[:-1]), number)
    except ValueError:
        print(f"Warning: Unhandled folder format: {folder_name}")
        return (folder_name, 0)
    
    return (folder_name, 0)

def analyze_evaluation_results():
    base_path = Path("/cluster/home/mgiulianelli/code/negotio2/multiturn-llm-training/evaluations/multi_game")
    
    # Dictionary to store results
    results = {
        'folder': [],
        'mean_reward': [],
        'mean_rent': [],
        'percentage_successful_episodes': [],
        'percentage_partially_unsuccessful_episodes': [],
        'std_reward': [],
    }
    
    # Iterate through all folders
    for folder in base_path.iterdir():
        if not folder.is_dir():
            continue
            
        # Find the JSON file in the folder
        json_files = list(folder.glob("*_results.json"))
        if not json_files:
            print(f"No results JSON found in {folder.name}")
            continue
            
        json_file = json_files[0]
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # Extract stats
            stats = data.get('stats', {})
            mean_reward = stats.get('mean_reward', None)
            std_reward = stats.get('std_reward', None)
            num_samples = stats.get('num_samples', 0)
            
            # Extract rent prices and calculate mean
            rent_prices = []
            num_successful_episodes = 0
            num_partially_unsuccessful_episodes = 0
            
            evaluations = data.get('evaluations', [])
            
            for episode in evaluations:
                if isinstance(episode, dict):
                    # Check if all values are N/A
                    all_na = all(str(v).upper() == 'N/A' for v in episode.values())
                    any_na = any(str(v).upper() == 'N/A' for v in episode.values())

                    print([str(v).upper() for v in episode.values()])
                    
                    if not all_na:
                        num_successful_episodes += 1
                    if any_na:
                        num_partially_unsuccessful_episodes += 1
                        
                    if 'rent' in episode:
                        if episode['rent'] == 'N/A':
                            rent_prices.append(0)
                        else:
                            try:
                                # Extract first number from rent string
                                numbers = re.findall(r'\d+', str(episode['rent']))
                                if numbers:
                                    rent_prices.append(int(numbers[0]))
                            except (ValueError, IndexError):
                                continue
                elif isinstance(episode, list):
                    # Handle nested list structure
                    for sub_episode in episode:
                        if isinstance(sub_episode, dict):
                            # Check if all values are N/A
                            all_na = all(str(v).upper() == 'N/A' for v in sub_episode.values())
                            any_na = any(str(v).upper() == 'N/A' for v in sub_episode.values())
                            
                            if not all_na:
                                num_successful_episodes += 1
                            if any_na:
                                num_partially_unsuccessful_episodes += 1
                                
                            if 'rent' in sub_episode:
                                if sub_episode['rent'] == 'N/A':
                                    rent_prices.append(0)
                                else:
                                    try:
                                        numbers = re.findall(r'\d+', str(sub_episode['rent']))
                                        if numbers:
                                            rent_prices.append(int(numbers[0]))
                                    except (ValueError, IndexError):
                                        continue
            
            # Calculate mean rent price
            mean_rent = np.mean(rent_prices) if rent_prices else None
            
            # Calculate percentages
            percentage_successful_episodes = (num_successful_episodes / num_samples) if num_samples > 0 else 0
            percentage_partially_unsuccessful_episodes = (num_partially_unsuccessful_episodes / num_samples) if num_samples > 0 else 0
            
            # Store results
            results['folder'].append(folder.name)
            results['mean_reward'].append(mean_reward)
            results['std_reward'].append(std_reward)
            results['mean_rent'].append(mean_rent)
            results['percentage_successful_episodes'].append(percentage_successful_episodes)
            results['percentage_partially_unsuccessful_episodes'].append(percentage_partially_unsuccessful_episodes)
            
        except Exception as e:
            print(f"Error processing {folder.name}: {str(e)}")
            # Add None values for failed processing
            results['folder'].append(folder.name)
            results['mean_reward'].append(None)
            results['std_reward'].append(None)
            results['mean_rent'].append(None)
            results['percentage_successful_episodes'].append(None)
            results['percentage_partially_unsuccessful_episodes'].append(None)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Add sorting columns
    df['method'] = df['folder'].apply(lambda x: extract_method_and_number(x)[0])
    df['number'] = df['folder'].apply(lambda x: extract_method_and_number(x)[1])
    
    # Print debug information
    print("\nDebug Information:")
    print("=" * 50)
    print("All folders found:")
    for folder in df['folder'].unique():
        method, number = extract_method_and_number(folder)
        print(f"Folder: {folder}, Method: {method}, Number: {number}")
    

    #Drop folder names and make method first column and number second column
    df = df.drop(columns=['folder'])
    df = df[['method', 'number', 'mean_reward', 'std_reward', 'mean_rent', 'percentage_successful_episodes', 'percentage_partially_unsuccessful_episodes']]
    
    # Sort the DataFrame
    df = df.sort_values(['method', 'number'])
    
    
    # Save to CSV
    output_path = base_path.parent / "evaluation_results_multi.csv"
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    return df

if __name__ == "__main__":
    results_df = analyze_evaluation_results()



