import glob
import os
from utils.visualization import display_images_from_folders
from scripts.train import SuperResolutionTrainer
from scripts.eval import SuperResolutionEvaluator


def find_config_files(pattern):
    """
    Find configuration files matching a specific pattern.
    """
    config_files = glob.glob(pattern)
    if not config_files:
        print(f"No configuration files found matching the pattern '{pattern}'")
    return config_files


def run_training(config_files):
    """
    Run training for each configuration file.
    """
    for config_file in config_files:
        try:
            print(f"Running Training with config: {config_file}")
            trainer = SuperResolutionTrainer(config_file)
            trainer.train()
        except Exception as e:
            print(f"Error during training with {config_file}: {e}")


def run_evaluation(config_files):
    """
    Run evaluation for each configuration file.
    """
    for config_file in config_files:
        try:
            print(f"Running Evaluation with config: {config_file}")
            evaluator = SuperResolutionEvaluator(config_file)
            evaluator.evaluate()
        except Exception as e:
            print(f"Error during evaluation with {config_file}: {e}")


def run_experiments():
    """
    Main function to run training and evaluation for super-resolution tasks.
    """
    config_pattern = "config/config_experiment*.yml"
    config_files = find_config_files(config_pattern)

    if config_files:
        print("\nStarting Experiments...\n")
        run_training(config_files)
        run_evaluation(config_files)
        print("\nExperiments completed.\n")
    else:
        print("No experiments to run.")


if __name__ == "__main__":
    run_experiments()

    # Display a batch comparison (per folder)
    try:
        display_images_from_folders("test_results")
    except Exception as e:
        print(f"Error displaying results: {e}")
