import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#%% --- 1. Configuration ---
FILE_PATH = './data/access_log_original.pkl'
USER_ID_COL = 'PERSON_ID'
RESOURCE_ID_COL = 'TARGET_NAME'
ACTION_COL = 'ACTION'
OUTPUT_DIR = './data'  # All processed files will be saved here

#%% --- 2. Load Data ---
# This section now uses your specific file path to load the DataFrame.
try:
    history_df = pd.read_pickle(FILE_PATH)
    print(f"Successfully loaded the original data with {len(history_df)} records.")
except Exception as e:
    print(f"An error occurred while loading the pickle file: {e}")
    exit()
print(f"Successfully loaded data from '{FILE_PATH}'.")
print(f"DataFrame shape: {history_df.shape}")
print("DataFrame head:")
print(history_df.head())

#%% --- 3. Calculate Event History Length for Each User ---
# We group by the user ID and then get the size of each group.
# This gives us a pandas Series where the index is the user ID
# and the value is the total number of their events.
user_event_lengths = history_df.groupby(USER_ID_COL).size()

print("\n--- Summary Statistics for User Event Lengths ---")
print(user_event_lengths.describe())

#%% --- 4. Plot the Histogram ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(6, 3))  # Use fig, ax for better control

# Create the histogram
# user_event_lengths as the first parameter is assigned to the X-axis
# kde=False ensures the Y-axis represents the actual “number of users”
sns.histplot(user_event_lengths, bins=50, kde=False, ax=ax)

# Set chart title and axis labels
# ax.set_title('Distribution of User Event History Lengths', fontsize=16)
ax.set_xlabel('Number of Events (Length)', fontsize=10)
ax.set_ylabel('Number of Users (Frequency)', fontsize=10)

# Set the Y-axis to logarithmic scale for better visualization of long-tail distributions
ax.set_yscale('log')

# Save and Display the Plot ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Use a new filename for the horizontal plot
output_path = os.path.join(OUTPUT_DIR, 'user_event_lengths_horizontal.pdf')

plt.savefig(output_path, format='pdf', bbox_inches='tight')
print(f"\nHorizontal plot successfully saved to: {output_path}")

plt.tight_layout()
plt.show()
