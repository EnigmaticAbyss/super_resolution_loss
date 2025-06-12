import glob
from utils.visualization import display_images_from_folders
from scripts.train import SuperResolutionTrainer
from scripts.eval import SuperResolutionEvaluator
from pathlib import Path

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


def run_evaluation(config_files= None):
    """
    Run evaluation for each configuration file.
    """
    if not config_files:
        handle_empty_config()
        return



    for config_file in config_files:
        try:
            print(f"Running Evaluation with config: {config_file}")
            evaluator = SuperResolutionEvaluator(config_file)
            evaluator.evaluate()
        except Exception as e:
            print(f"Error during evaluation with {config_file}: {e}")


def handle_empty_config():
    """
    Handle the case when no configuration files are provided.
    """
    print("Handling the empty folder of configs through saved models")
    main_dir = Path(__file__).resolve().parent
    saved_model_dir = main_dir / "saved_models"

    saved_models = [f for f in saved_model_dir.iterdir() if f.is_file()]

    for saved_model in saved_models:
        print(f"Processing model: {saved_model}")
        evaluator = SuperResolutionEvaluator(saved_model)
        evaluator.evaluate()
        # Your evaluation logic here
        

def run_experiments():
    """
    Main function to run training and evaluation for super-resolution tasks.
    """
    config_pattern = "config/config_experiment*.yml"
    config_files = find_config_files(config_pattern)
    # print("config files")
    # print(config_files)
    if config_files:
        print("\nStarting Experiments...\n")
        # run_training(config_files)
        run_evaluation(config_files)
        print("\nExperiments completed.\n")

    else:
        print("No experiments to run.")
        """run with no config"""
        run_evaluation()


if __name__ == "__main__":
    run_experiments()

    # Display a batch comparison (per folder)
    try:
        display_images_from_folders("test_results")
    except Exception as e:
        
        print(f"Error displaying results: {e}")
