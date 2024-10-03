import pandas as pd

#input and output files
input_file = 'indiana_top_10.csv'
output_file = 'cleaned_indiana_top_10.csv'

#read the file and treat "-" as a missing value
df = pd.read_csv(input_file, na_values='-')

#save the cleaned file
df.to_csv(output_file, index=False)
print(f'Cleaned CSV file saved to {output_file}')
