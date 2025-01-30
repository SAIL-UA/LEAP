import os
import pandas as pd
import random

def generate_ground_truth(demographic_file, parent_folder):
    """
    Generates ground truth labels for each participant based on demographic data.

    Parameters:
    - demographic_file (str): Path to the Excel file containing demographic data.
    - parent_folder (str): Path to the parent folder containing participant subfolders.

    Returns:
    - dict: Summary statistics including counts and missing data percentages.
    """

    # Load the demographic file
    demographic_data = pd.read_excel(demographic_file)

    # Initialize participant lists
    complete_participants = []
    no_match_participants = []

    # Initialize counters for labels
    stats = {
        "Total Left": 0, "Total Right": 0, "Total Healthy": 0,
        "Total Male": 0, "Total Female": 0,
        "Male Left": 0, "Male Right": 0, "Male Healthy": 0,
        "Female Left": 0, "Female Right": 0, "Female Healthy": 0
    }

    # Iterate over participant subfolders
    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)
        if os.path.isdir(subfolder_path):
            # Find the corresponding row in demographic data
            patient_data = demographic_data[demographic_data['Patient ID'] == subfolder]
            
            if not patient_data.empty:
                # Extract required information
                involved_limb = patient_data['Involved Limb'].values[0] if pd.notna(patient_data['Involved Limb'].values[0]) else 'Healthy'
                sex = patient_data['Sex'].values[0] if pd.notna(patient_data['Sex'].values[0]) else random.choice(['M', 'F'])
                
                # Standardize labels
                involved_limb_text = (
                    'Left injured' if 'left' in involved_limb.lower() else 
                    'Right injured' if 'right' in involved_limb.lower() else 
                    'Healthy'
                )
                sex_text = 'Male' if sex.lower() == 'm' else 'Female' if sex.lower() == 'f' else 'Unknown'
                
                # Update statistics
                stats[f"Total {involved_limb_text.split()[0]}"] += 1
                stats[f"Total {sex_text}"] += 1
                if involved_limb_text != 'Healthy':
                    stats[f"{sex_text} {involved_limb_text.split()[0]}"] += 1
                else:
                    stats[f"{sex_text} Healthy"] += 1
                
                # Write Ground_Truth.txt
                info_file_path = os.path.join(subfolder_path, 'Ground_Truth.txt')
                if os.path.exists(info_file_path):
                    os.remove(info_file_path)
                with open(info_file_path, 'w') as file:
                    file.write(f"{involved_limb_text} {sex_text}")

                complete_participants.append(subfolder)
            else:
                no_match_participants.append(subfolder)

    # Compute missing data percentage
    total_participants = len(complete_participants) + len(no_match_participants)
    missing_percentage = (len(no_match_participants) / total_participants) * 100 if total_participants > 0 else 0

    # Return statistics and participant lists
    return {
        "Complete Participants": complete_participants,
        "No Match Participants": no_match_participants,
        "Missing Percentage": missing_percentage,
        **stats
    }
