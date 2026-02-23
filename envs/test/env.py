from datasets import Dataset


class TestEnv:
    def __init__(self, **kwargs):
        """
        A minimal test environment with the same interface as NegotiationEnv.
        Used for quick sanity-check runs without any real game logic.
        """
        pass

    def create_dataset(self, size: int = 100) -> Dataset:
        """
        Create a simple dataset of identical prompts.
        """
        prompt = (
            "Please generate a random number between 0 and 100. "
            "Only respond with the number, nothing else."
        )
        prompt_2 = (
            "Please generate a random number between 0 and 100. "
            "Only respond with the number, nothing else."
        )
        samples = [
            {
                "prompt": prompt,
                "prompt_2": prompt_2,
                "starting_agent": False,
                "negotiation_role": 0,
                "game_config": {},
            }
            for _ in range(size)
        ]
        return Dataset.from_list(samples)

    def get_reward_functions(self):
        """
        Returns a list of trivial reward functions (penalises deviation from
        a target completion length of 20 characters).
        """

        def test_reward_len(prompts, completions,**kwargs):

            rewards = []
            for i, messages in enumerate(completions):
                total_reward = 500.0
                last_user_number = 0
              
                for message in messages:
                    if message["role"] == "user":
                        try:
                            last_user_number = int(message["content"])
                        except Exception as e:
                            print(f"Error parsing user number: {e}")
                            last_user_number = 0
                    if message["role"] == "assistant":
                        try:
                            last_assistant_number = int(message["content"])
                            difference = abs(last_assistant_number - (100 - last_user_number))
                        except Exception as e:
                            print(f"Error parsing assistant number: {e}")
                            last_assistant_number = 0
                            difference = 100
                        total_reward -= float(difference)
                
                total_reward = total_reward / float(len(messages))

                rewards.append(total_reward)    

            print("Rewards:", rewards)
            return rewards

        return [test_reward_len]
