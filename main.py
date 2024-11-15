import glob
import os
import subprocess
from utils.visualization import display_images_from_folders

def run_experiments():

    # List all configuration files that follow the naming convention "config_experiment*.yaml"
    config_files = glob.glob("config/config_experiment*.yml")

    # Check if there are any config files found
    if not config_files:
        print("No configuration files found matching the pattern 'config/config_experiment*.yaml'")
    else:
        
        
        
        # Run each experiment
        for config_file in config_files:
            print(f"Running experiment with config: {config_file}")
            try:
           
             
                result_train = subprocess.run(
                    ["python", "scripts/train.py", "--config", config_file],
                    capture_output=True,
                    text=True,
                    check=True
                )
             
                print(f"Train experiment with {config_file} completed successfully.\nOutput:\n{result_train.stdout}")
                
            except subprocess.CalledProcessError as e:
                print(f"Train with {config_file} failed.\nError:\n{e.stderr}")
                
                
        # Test each experiment
        for config_file in config_files:
            print(f"Running Test with config: {config_file}")
            try:
           
             
                result_eval = subprocess.run(
                    ["python", "scripts/eval.py", "--eval", config_file],
                    capture_output=True,
                    text=True,
                    check=True
                )
             
                print(f"Train experiment with {config_file} completed successfully.\nOutput:\n{result_eval.stdout}")
                
            except subprocess.CalledProcessError as e:
                print(f"Test with {config_file} failed.\nError:\n{e.stderr}")    
    
    
    


# Main execution block
if __name__ == "__main__":
    # run_experiments()
 
    # Display a batch comparison (per folder)
    display_images_from_folders("test_results")
    
    
    
    
    
    
