import attr
from attr import define, field
from typing import Any
import json
import re
import os
from envs.negotiation.games import Game




@define
class Evaluator:
    game: Game = attr.ib()
    model: Any = None
    game_type: str = "generic-rental-agreement"


    def evaluate(self, trajectory, starting_agent=0, get_payoffs=False): 

        system_message = self.get_system_msg()


        full_input_prompt = self.get_full_input_prompt(system_message, trajectory, starting_agent)

        max_retries = 1  # Maximum number of retry attempts
        attempt = 0
        result = None

        while attempt < max_retries and result is None:
            try:
                response = self.model(full_input_prompt)  # Generate model response

                agreements = self.extract_evaluation(response)  # Try to parse the response

                if agreements is not None and self.all_required_keys_present(agreements):

                    result = agreements
                    result["response"] = response

                    pay_off_tables = {"Agent 1": [], "Agent 2": []}
                    for agent_idx, agent in enumerate(["Agent 1", "Agent 2"]):
                        #Add the payoff_tables
                        for issue in self.game.issues:
                            payoff_table = {
                                "issue": issue.name,
                                "payoff_labels": issue.payoff_labels[agent_idx].tolist() if hasattr(issue.payoff_labels[agent_idx], 'tolist') else issue.payoff_labels[agent_idx],
                                "payoff_values": issue.payoffs[agent_idx].tolist() if hasattr(issue.payoffs[agent_idx], 'tolist') else issue.payoffs[agent_idx]
                            }
                            pay_off_tables[agent].append(payoff_table)

                    result["pay_off_tables"] = pay_off_tables


                    if get_payoffs:
                        payoffs = self.get_payoffs(agreements)  # Extract the payoff information
                        result["payoffs"] = payoffs

                    break  # Exit loop if a valid result is obtained

                

            except Exception as e:
                print(f"Error during evaluation attempt {attempt + 1}: {e}")

            attempt += 1

        if result is None:
            print("All attempts to evaluate failed. Returning None.")
        else:
            pass

        return result
    

    def get_payoffs(self, agreements):
        issues = self.game.issues

        payoffs = {
            "Agent 1": 0,
            "Agent 2": 0
        }

        for i, agent in enumerate(["Agent 1", "Agent 2"]):

            for issue_name, value in agreements.items():

                if value is None or value == "N/A":
                    continue
                
                issue = next((issue for issue in issues if issue.name == issue_name), None)

                #Search for issue in game
                if issue is None:
                    continue
                
                payoff_labels = issue.payoff_labels[i]
                payoff_values = issue.payoffs[i]

                # Extract numeric value from agreement
                numeric_value = self.extract_numeric_value(value)
                if numeric_value is None:
                    print(f"Could not extract numeric value from agreement value '{value}' for {agent}.")
                    continue

                # Calculate payoff using interpolation
                payoff = self.interpolate_payoff(numeric_value, payoff_labels, payoff_values)
                if payoff is None:
                    print(f"Could not calculate payoff for value '{numeric_value}' in issue '{issue_name}' for {agent}.")
                    continue

                # Add to total payoff for the agent
                payoffs[agent] += payoff

                

        return payoffs
    


    def extract_numeric_value(self, label):
        """
        Extracts the numeric value from a label string like '$750', '750 CHF', etc.
        Returns None if no numeric value is found.
        """
        match = re.search(r'\d+\.?\d*', label)

        return float(match.group()) if match else None

    def interpolate_payoff(self, value, labels, payoffs):
        """
        Interpolates the payoff for a given value between numeric labels.
        """
        numeric_labels = [self.extract_numeric_value(label) for label in labels]

        for i in range(len(numeric_labels) - 1):
            if numeric_labels[i] <= value <= numeric_labels[i + 1]:
                # Linear interpolation formula
                t = (value - numeric_labels[i]) / (numeric_labels[i + 1] - numeric_labels[i])
                return payoffs[i] + t * (payoffs[i + 1] - payoffs[i])
        
        print(f"Value {value} is out of the range of provided labels.")
        return None  # Return None if value is out of range
    
    def change_game(self):
        #Implement this function to change the game during evaluation
        pass

    def all_required_keys_present(self, result):
        # required_keys = ["rent"]
        required_keys = [issue.name for issue in self.game.issues]
        for key in required_keys:
            if key not in result:
                print(f"Key '{key}' not found in the evaluation result.")
                return False
        return True
    

    def get_system_msg(self):

        single_issue = len(self.game.issues) == 1

        # Get the directory where this file is located (evaluator/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        try:
            if self.game_type == "generic-rental-agreement":
                file_path = os.path.join(current_dir, 'evaluation_outcome.txt')
                with open(file_path, 'r', encoding='utf-8') as file:
                    system_message = file.read()
                return system_message
            elif self.game_type == "multi-game" or self.game_type == "out-of-domain":
                if single_issue:
                    issue_name = self.game.issues[0].name
                    payoff_example = self.game.issues[0].payoff_labels[0][4]
                    file_path = os.path.join(current_dir, 'evaluation_outcome_single.txt')
                    with open(file_path, 'r', encoding='utf-8') as file:
                        system_message = file.read()

                        system_message = system_message.replace("ISSUE_NAME", issue_name)
                        system_message = system_message.replace("EXAMPLE", payoff_example)
                    return system_message
                else:
                    issue_names = [issue.name for issue in self.game.issues]
                    payoff_examples = [issue.payoff_labels[0][4] for issue in self.game.issues]

                    file_path = os.path.join(current_dir, 'evaluation_outcome_multi.txt')
                    with open(file_path, 'r', encoding='utf-8') as file:
                        system_message = file.read()
                        system_message = system_message.replace("ISSUE1_NAME", issue_names[0])
                        system_message = system_message.replace("ISSUE2_NAME", issue_names[1])
                        system_message = system_message.replace("EXAMPLE1", payoff_examples[0])
                        system_message = system_message.replace("EXAMPLE2", payoff_examples[1])
                    return system_message

        except FileNotFoundError:
            print("System message file not found.")
            return ''

    

    def get_full_input_prompt(self, system_message, trajectory, starting_agent=0):
        full_prompt = [{'role': 'system', 'content': system_message}]

        user_content = "\n\n Negotiation Conversation: \n\n"

        if starting_agent == 0:
            roles ={
                "assistant": self.game.parties[0],
                "user": self.game.parties[1],
                "system": "System"
            }
        else:
            roles ={
                "assistant": self.game.parties[1],
                "user": self.game.parties[0],
                "system": "System"
            }

        for message in trajectory:
            # Map the roles to labels
            role_label = roles[message['role']]

            if role_label != 'System':
                new_content = f"{role_label}: {message['content']}"
                user_content += new_content + '\n'

        full_prompt.append({'role': 'user', 'content': user_content})

        return full_prompt



    def extract_evaluation(self, response):
        # Handle the model's response
           
        start = response.find("{")  # Locate the first opening curly brace
        end = response.rfind("}")  # Locate the last closing curly brace
        if start != -1 and end != -1 and start < end:
            try:
                # Extract substring and attempt to parse it as JSON
                json_text = response[start:end + 1]
                return json.loads(json_text)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
        return None






