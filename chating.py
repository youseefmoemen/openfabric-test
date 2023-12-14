import requests

# Define the API URL and headers
url = "http://localhost:5500/execution"
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
}

# Define the data to send
#data = {"text": ["What is the capital of Egypt?", "How many elements in the periodic table?", "What is the normal distribution"]}
data = {"text": ["What is atomic number", "What is the capital of France", "What is DNA"]}
# Send a POST request and handle the response
response = requests.post(url, headers=headers, json=data)

# Check for successful response
if response.status_code == 200:
    # Get the response data
    data = response.json()
    print(f"Response: {data}")
else:
    print(f"Error: {response.status_code} - {response.reason}")
