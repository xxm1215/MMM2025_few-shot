# Define the function to extract the required values from the files
import re

def extract_values(file_contents):
    """Extract accuracy and f1-score values from file contents."""
    accuracy_values = []
    f1_score_values = []
    for content in file_contents:
        # Extract accuracy
        accuracy_match = re.search(r'accuracy\s*(\d+\.\d+)', content)
        if accuracy_match:
            accuracy_values.append(float(accuracy_match.group(1)))
        
        # Extract f1-score (macro average)
        f1_score_match = re.search(r'macro avg\s*\d+\.\d+\s*\d+\.\d+\s*(\d+\.\d+)', content)
        if f1_score_match:
            f1_score_values.append(float(f1_score_match.group(1)))
    
    return accuracy_values, f1_score_values

# Read the contents of the files
filenames = [
    'E:/code/20240330_ccl/prompt/results/poli/few/poli_50st_1sd_1.0al_false.txt',
    'E:/code/20240330_ccl/prompt/results/poli/few/poli_50st_2sd_1.0al_false.txt',
    'E:/code/20240330_ccl/prompt/results/poli/few/poli_50st_3sd_1.0al_false.txt',
    'E:/code/20240330_ccl/prompt/results/poli/few/poli_50st_4sd_1.0al_false.txt',
    'E:/code/20240330_ccl/prompt/results/poli/few/poli_50st_5sd_1.0al_false.txt'
]

file_contents = []
for filename in filenames:
    with open(filename, 'r') as file:
        file_contents.append(file.read())

# Extract the accuracy and f1-score values
accuracy_values, f1_score_values = extract_values(file_contents)

print("accuracy_values",accuracy_values)
print("f1_score_values",f1_score_values)

# Remove the minimum and maximum values and calculate the average for accuracy and f1-score
def calculate_adjusted_average(values):
    """Calculate the average after removing the minimum and maximum values."""
    if len(values) > 2:
        return sum(values) / len(values)
    return sum(sorted(values)[1:-1]) / (len(values) - 2)  # Correct calculation when more than two values

adjusted_accuracy_average = calculate_adjusted_average(accuracy_values)
adjusted_f1_score_average = calculate_adjusted_average(f1_score_values)

print("acc-f1",adjusted_accuracy_average)
print("macro_f1",adjusted_f1_score_average)

# adjusted_accuracy_average, adjusted_f1_score_average
