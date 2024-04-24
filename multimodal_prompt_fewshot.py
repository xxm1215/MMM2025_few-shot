import os
import pandas as pd

# Function to calculate mean value excluding max and min
def calculate_mean(values):
    if len(values) <= 2:
        return None
    max_value = max(values)
    min_value = min(values)
    values.remove(max_value)
    values.remove(min_value)
    return sum(values) / len(values)

# List all xlsx files in the folder
folder_path = "/mnt/qust_521_big_2/20240330_CCL/prompt/results_excel_4heads/goss/full"
xlsx_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

# Initialize lists to store calculated values
names = []
mean_accs = []
mean_macro_f1s = []

# Iterate through each file
for file_name in xlsx_files:
    # Read data from Excel file
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_excel(file_path)
    
    # Calculate mean values
    mean_acc = calculate_mean(df['acc'].tolist())
    mean_macro_f1 = calculate_mean(df['macro_f1'].tolist())
    
    # Store results
    name = os.path.splitext(file_name)[0]  # Remove .xlsx extension
    names.append(name)
    mean_accs.append(mean_acc)
    mean_macro_f1s.append(mean_macro_f1)

# Create DataFrame from lists
result_df = pd.DataFrame({'name': names, 'acc': mean_accs, 'macro_f1': mean_macro_f1s})

# 最终的结果。
result_df.to_excel('result_goss_full.xlsx', index=False)

