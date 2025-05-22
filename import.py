import json
import csv

# Input file path
input_file = '/Users/amanueltefera/Downloads/appraisals_dataset.json'

# Output file path
output_file = '/Users/amanueltefera/Desktop/combined_appraisals.csv'

# Load the JSON
with open(input_file, 'r') as f:
    data = json.load(f)

# Get sample keys
subject_keys = list(data['appraisals'][0]['subject'].keys())
comp_keys = list(data['appraisals'][0]['comps'][0].keys())
property_keys = list(data['appraisals'][0]['properties'][0].keys())

# Build CSV fieldnames
fieldnames = ['orderID'] + ['subject_' + k for k in subject_keys] + \
             ['comp_' + k for k in comp_keys] + ['property_' + k for k in property_keys]

# Write to one CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for appraisal in data['appraisals']:
        order_id = appraisal['orderID']
        subject = {'subject_' + k: v for k, v in appraisal['subject'].items()}

        # Pair each comp with each property
        for comp in appraisal['comps']:
            comp_data = {'comp_' + k: v for k, v in comp.items()}
            for prop in appraisal['properties']:
                prop_data = {'property_' + k: v for k, v in prop.items()}
                row = {'orderID': order_id}
                row.update(subject)
                row.update(comp_data)
                row.update(prop_data)
                writer.writerow(row)

print("✅ Combined CSV created at your Desktop!")
