import glob
import os
import subprocess

def run_experiments():
    # List all configuration files that follow the naming convention "config_experiment*.yaml"
    config_files = glob.glob("config/config_experiment*.yaml")

    # Check if there are any config files found
    if not config_files:
        print("No configuration files found matching the pattern 'config/config_experiment*.yaml'")
    else:
        # Run each experiment
        for config_file in config_files:
            print(f"Running experiment with config: {config_file}")
            try:
                # Use subprocess to capture output and handle errors
                result = subprocess.run(
                    ["python", "scripts/train.py", "--config", config_file],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"Experiment with {config_file} completed successfully.\nOutput:\n{result.stdout}")
            except subprocess.CalledProcessError as e:
                print(f"Experiment with {config_file} failed.\nError:\n{e.stderr}")

# Main execution block
if __name__ == "__main__":
    run_experiments()
