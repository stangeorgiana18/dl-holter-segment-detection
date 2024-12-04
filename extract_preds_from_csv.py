import json

# Load the JSON results
with open('test_results.json', 'r') as f:
    results = json.load(f)

# Specify the target file
target_file = "JS00823_signals.csv"

# Prepare the filtered output
filtered_results = {
    "file": {},
    "predictions": {}
}

# Search for the target file
for entry in results:
    if entry["file"] == target_file:
        filtered_results["file"] = entry["file"]
        filtered_results["predictions"] = entry["predictions"]
        break

# Save or display the filtered results
with open('filtered_results.json', 'w') as f:
    json.dump(filtered_results, f, indent=4)

print(f"Filtered results saved to 'filtered_results.json'")
