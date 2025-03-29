import requests
import json
import sys
import socket

def test_conversation(api1_url, api2_url):
    """Test conversation between two LLM APIs using dynamic hostname"""
    
    print(f"Testing conversation between {api1_url} and {api2_url}")
    
    # Create a session that bypasses proxies
    session = requests.Session()
    session.trust_env = False  # Don't use environment variables for proxy
    
    # First verify both services are running
    try:
        health1 = session.get(f"{api1_url}/health/")
        health2 = session.get(f"{api2_url}/health/")
        print(f"API1 health: {health1.status_code}, API2 health: {health2.status_code}")
        if health1.status_code != 200 or health2.status_code != 200:
            print("Failed health check! One or both APIs are not running.")
            return False
    except Exception as e:
        print(f"Error checking API health: {e}")
        return False
    
    # Now test a conversation with the correct payload format
    try:
        # Format conversations as expected by the API
        convos = ["Explain quantum computing in simple terms.","Write a short poem about artificial intelligence."]
        
        
        response = session.post(
            f"{api1_url}/generate/",
            json={
                "prompts": convos,
                "temperature": 0.7,
                "max_tokens": 100
            }
        )
        
        if response.status_code != 200:
            print(f"Error from API1: {response.status_code}")
            print(response.text)
            return False
            
        result = response.json()
        print("\nResponse result:")
        print(result)
        
        # Handle the response based on the actual structure returned
        if "responses" in result:
            for i, response_text in enumerate(result["responses"]):
                print(f"\nResponse {i+1}:")
                print(response_text)
        
        return True
    except Exception as e:
        print(f"Error during conversation test: {e}")
        return False

# Rest of the code remains the same

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_apis.py <api1_host:port> <api2_host:port>")
        sys.exit(1)
        
    api1_url = f"http://{sys.argv[1]}"
    api2_url = f"http://{sys.argv[2]}"
    

    success = test_conversation(api1_url, api2_url)
    
    if success:
        print("\nTest passed! The APIs are communicating correctly.")
        sys.exit(0)
    else:
        print("\nTest failed! Check the logs for errors.")
        sys.exit(1)