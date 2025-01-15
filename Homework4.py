import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set global font size
plt.rcParams.update({'font.size': 14})

save_path = "C:/Users/jojor/OneDrive/Desktop/Uni/Master/WS2425/DSSS/HW4/"

# Load the CSV file
df = pd.read_csv("C:/Users/jojor/OneDrive/Desktop/Uni/Master/WS2425/DSSS/HW4/census_income_dataset.csv")

# Set style for the plots
sns.set(style="whitegrid")

# Plot a) Age Distribution of respondents
plt.figure(figsize=(10, 6))
sns.histplot(df['AGE'], bins=20, kde=True, color="skyblue")
plt.title('Age Distribution of Respondents', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.savefig(save_path + "age_distribution.svg", format='svg', bbox_inches='tight')  # Save as SVG
plt.show()

# Plot b) Frequency of each Relationship Status
plt.figure(figsize=(14, 6))
sns.countplot(data=df, x='RELATIONSHIP', hue='RELATIONSHIP', dodge=False, palette="viridis", legend=False)
plt.title('Frequency of Relationship Status', fontsize=16)
plt.xlabel('Relationship Status', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=60, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(save_path+"relationship_status_frequency.svg", format='svg', bbox_inches='tight')  # Save as SVG
plt.show()


# Plot c) Stacked Bar Chart for Salary by Educational Level

# Convert SALARY to binary: 0 for <=50K, 1 for >50K
df['SALARY_BINARY'] = df['SALARY'].apply(lambda x: 0 if x == ' <=50K' else 1)

# Group by education level and salary binary, then unstack and fill missing values with 0
edu_salary_counts = df.groupby(['EDUCATION', 'SALARY_BINARY']).size().unstack(fill_value=0)

# Rename columns for clarity: '0' becomes '<=50K' and '1' becomes '>50K'
edu_salary_counts.columns = ['<=50K', '>50K']

# Calculate the percentage of each salary category within each education level
edu_salary_percentage = edu_salary_counts.div(edu_salary_counts.sum(axis=1), axis=0) * 100

# Plot the stacked bar chart
edu_salary_percentage.plot(kind='bar', stacked=True, figsize=(16, 5), color=['lightblue', 'salmon'])
plt.title('Percentage of Salary Distribution by Education Level', fontsize=16)
plt.xlabel('Education Level', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.legend(
    title="Salary",
    labels=["<=50K", ">50K"],
    loc="upper left",
    bbox_to_anchor=(1.05, 1),
    fontsize=12,
    title_fontsize=14
)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(save_path + "salary_distribution_by_education.svg", format='svg', bbox_inches='tight')  # Save as SVG
plt.show()
