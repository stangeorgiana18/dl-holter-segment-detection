import json

# Load JSON file
with open('test_results.json', 'r') as f:
    data = json.load(f)

# Flatten predictions
flattened_results = []
for entry in data:
    file_name = entry['file']
    predictions = entry['predictions']
    
    for lead, probabilities in predictions.items():
        flattened_results.append({
            "file": file_name,
            "lead": lead,
            "probabilities": probabilities[0]
        })

# Save the flattened JSON
with open('flattened_test_results.json', 'w') as f:
    json.dump(flattened_results, f, indent=4)

print("Flattened JSON saved to flattened_test_results.json")
