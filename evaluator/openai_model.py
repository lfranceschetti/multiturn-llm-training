from retry import retry
from openai import OpenAI
from attr import define, field
import requests

import sys
from helpers.model_utils import ChatModel


@define
class OpenAIModel(ChatModel):
    model_provider: str = field(default='openai')
    model_name: str = field(default='gpt-4o-mini')
    role_mapping = field(default={'role': 'role', 'content': 'content', 'assistant': 'assistant', 'user': 'user',
                                  'system': 'system'})
    

    def __attrs_post_init__(self):
        """
        This method is called after the instance has been initialized. 
        It initializes the model.
        """
        super().__attrs_post_init__()  # Ensures the parent logic is executed first
        
        self.model = OpenAI(api_key=self.model_key)
    
    # @retry(Exception, tries=2, delay=2, backoff=2)
    def _generate(self, data):

        # headers = {
        #     'Content-Type': 'application/json',
        #     'Authorization': f'Bearer {self.model_key}'
        # }

        # API_BASE_URL='https://litellm.sph-prod.ethz.ch/'
        # url = API_BASE_URL+'chat/completions'

        # payload = {
        #     "model": self.model_name,
        #     "messages": data,
        #     "temperature": self.temperature,
        # }

        # response = requests.post(url, headers=headers, json=payload)
        # response = response.json()
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=data,
            temperature=self.temperature,
        )

        ai_msg = self._postprocess(response)

        return ai_msg

   

    def _postprocess(self, data):
        content = ""
        try:
            # Log the full response for debugging
            
            # Handle both dictionary and ChatCompletion object responses
            if isinstance(data, dict):
                if "choices" not in data:
                    print(f"Response missing 'choices' field. Full response: {data}")
                    return content
                    
                if not data["choices"]:
                    print("Empty choices array in response")
                    return content
                    
                content = data["choices"][0]["message"]["content"]
            else:
                # Handle ChatCompletion object
                content = data.choices[0].message.content
                
        except Exception as e:
            print(f'[error] failed to generate response - {e}')
            print(f'Response data: {data}')

        return content

if __name__ == "__main__":
    model_ = OpenAIModel(model_provider='azure')
