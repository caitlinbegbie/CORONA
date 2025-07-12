# title: aavsoconversion.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 


# txt to csv:
def to_csv(input_folder_path, output_folder_path):
    converted_files_list = []
    os.makedirs(output_folder_path, exist_ok=True)
    
    for file_name in os.listdir(input_folder_path):
        current_file_path = os.path.join(input_folder_path, file_name)
       
        if os.path.isfile(current_file_path) and (file_name.endswith('.txt') or file_name.endswith('.csv')):
            try:
                df = pd.read_csv(current_file_path, comment='#', names=["JD", "Magnitude", "Uncertainty", "HQuncertainty",
                                                               "Band", "Observer Code", "Comment Code(s)",
                                                                "Comp Star 1", "Comp Star 2", "Charts", "Comments",
                                                               "Transformed", "Airmass", "Validation Flag", "Cmag",
                                                                "Kmag", "HJD", "Star Name", "Observer Affiliation",
                                                               "Measurement Method", "Grouping method", "ADS Reference",
                                                               "Digitizer", "Credit"])
               
    
                base_filename = os.path.splitext(file_name)[0]
                new_filename = f"{base_filename}.csv"
                new_file_path = os.path.join(output_folder_path, new_filename)
                df.to_csv(new_file_path, index=False)
                converted_files_list.append(new_file_path)
                
                print(f"Successfully converted '{file_name}' to '{new_filename}'")
            except Exception as e:
                
                print(f"Error converting '{file_name}': {e}")
        elif os.path.isfile(current_file_path):
            print(f"Skipping '{file_name}': Not a .txt or .csv file.")
        else:
            print(f"Skipping '{file_name}': Is a directory.")

    return converted_files_list

