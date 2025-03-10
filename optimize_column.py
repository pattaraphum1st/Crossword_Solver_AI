import pandas as pd
import unidecode

# Load your CSV file
df = pd.read_csv("C:/Users/USER/Downloads/optimized_times.csv")

# Clean the text
df['optimized_clue'] = df['optimized_clue'].apply(lambda x: unidecode.unidecode(str(x)))

# Save the cleaned file
df.to_csv("cleaned_optimized_times.csv", index=False)