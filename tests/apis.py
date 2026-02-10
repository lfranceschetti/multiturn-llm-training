import requests
import json
import sys
import socket

def test_conversation(api0_url, api1_url):
    """Test conversation between two LLM APIs using dynamic hostname"""
    
    print(f"Testing conversation between {api0_url} and {api1_url}")
    
    # Create a session that bypasses proxies
    session = requests.Session()
    session.trust_env = False  # Don't use environment variables for proxy
    
    # First verify both services are running
    try:
        health1 = session.get(f"{api0_url}/health/")
        health2 = session.get(f"{api1_url}/health/")
        print(f"API0 health: {health1.status_code}, API1 health: {health2.status_code}")
        if health1.status_code != 200 or health2.status_code != 200:
            print("Failed health check! One or both APIs are not running.")
            return False
    except Exception as e:
        print(f"Error checking API health: {e}")
        return False
    
    # Now test a conversation with the correct payload format
    try:
        # Format conversations as expected by the API
        convos = ["Lets count together. You should just say the next following number as a word and nothing else: Example: 'One' and then the next person says 'Two' and so on. I will start: One",
                "Keep up a brief conversation about the weather. I will start: It's a beautiful day today."]
        
        response = session.post(
            f"{api0_url}/generate/",
            json={
                "prompts": convos,
                "prompts_2": convos,
                "temperature": 0.7,
                "max_completion_length": 100
            }
        )
        
        if response.status_code != 200:
            print(f"Error from API0: {response.status_code}")
            print(response.text)
            return False
            
        result = response.json()
        print("\nResponse result:")
        print(result)
        
        # Handle the response based on the new structure
        if "conversations" in result:
            for i, conversation in enumerate(result["conversations"]):
                print(f"\nConversation {i+1}:")
                for j, message in enumerate(conversation):
                    print(f"  {message['role']}: {message['content']}")
            
            # If you want to check the token data
            if "token_ids" in result and "assistant_masks" in result:
                print("\nToken data:")
                print(f"Number of conversations with token data: {len(result['token_ids'])}")
                
                for i in range(len(result['token_ids'])):
                    assistant_tokens_count = sum(sum(mask) for mask in result['assistant_masks'][i])
                    print(f"Conversation {i+1} assistant tokens: {assistant_tokens_count}")
        
        return True
    except Exception as e:
        print(f"Error during conversation test: {e}")
        return False

# Rest of the code remains the same

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_apis.py <api1_host:port> <api2_host:port>")
        sys.exit(1)
        
    api0_url = f"http://{sys.argv[1]}"
    api1_url = f"http://{sys.argv[2]}"
    

    success = test_conversation(api0_url, api1_url)
    
    if success:
        print("\nTest passed! The APIs are communicating correctly.")
        sys.exit(0)
    else:
        print("\nTest failed! Check the logs for errors.")
        sys.exit(1)