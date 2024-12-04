import json

# Load the JSON file
with open('test_results.json', 'r') as f:
    results = json.load(f)

# Extract the 'file' dictionary from the root
if "file" in results:
    file_dict = results["file"]
else:
    raise KeyError("The JSON structure does not contain a 'file' key.")

# Get the last 50 entries from the 'file' dictionary
last_50_files = list(file_dict.items())[-50:]

# Display the last 50 files
for idx, file_name in last_50_files:
    print(f"Index: {idx}, File: {file_name}")
