# License: MIT
# This example demonstrates compressor visualization features.
# Supports both mock data and user-provided config_space.json / history files.
#
# Usage:
#   python visualization_example.py                                     # Use mock data
#   python visualization_example.py --config-space /path/to/config.json # Use custom config space
#   python visualization_example.py --history /path/to/history.json     # Use custom history
#   python visualization_example.py --mode basic                        # Static HTML mode

import os
import json
import argparse
import numpy as np
from glob import glob

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from openbox.utils.history import History, Observation
from openbox.utils.constants import SUCCESS
from openbox.compressor import Compressor, SHAPDimensionStep, BoundaryRangeStep
from openbox.compressor.api.compress_api import create_config_space_from_dict, load_history_from_dict


# Default Search Space (used when --config-space not provided)
def create_default_config_space():
    cs = ConfigurationSpace(seed=42)
    cs.add_hyperparameters([
        UniformFloatHyperparameter('learning_rate', 1e-5, 1e-1, default_value=1e-3, log=True),
        UniformFloatHyperparameter('momentum', 0.1, 0.99, default_value=0.9),
        UniformFloatHyperparameter('weight_decay', 1e-6, 1e-2, default_value=1e-4, log=True),
        UniformIntegerHyperparameter('batch_size', 16, 512, default_value=64, log=True),
        UniformIntegerHyperparameter('num_layers', 1, 8, default_value=3),
        UniformIntegerHyperparameter('hidden_size', 32, 1024, default_value=256, log=True),
        UniformFloatHyperparameter('dropout', 0.0, 0.5, default_value=0.1),
        UniformFloatHyperparameter('warmup_ratio', 0.0, 0.2, default_value=0.1),
        UniformFloatHyperparameter('beta1', 0.8, 0.99, default_value=0.9),
        UniformFloatHyperparameter('beta2', 0.9, 0.999, default_value=0.999),
        UniformFloatHyperparameter('epsilon', 1e-9, 1e-6, default_value=1e-8, log=True),
        UniformIntegerHyperparameter('epochs', 10, 200, default_value=50),
    ])
    return cs


# Load config space from JSON file
def load_config_space(config_path: str) -> ConfigurationSpace:
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return create_config_space_from_dict(config_dict)


# Load history from JSON file(s)
def load_history(history_path: str, config_space: ConfigurationSpace) -> History:
    """Load history from a JSON file or directory containing JSON files."""
    if os.path.isdir(history_path):
        # Find all JSON files in directory
        json_files = glob(os.path.join(history_path, '*.json'))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {history_path}")
        history_path = json_files[0]  # Use first file
        print(f"Loading history from: {history_path}")
    
    with open(history_path, 'r') as f:
        data = json.load(f)
    
    # Check if it's a list (direct observations) or dict (History format)
    if isinstance(data, list):
        return load_history_from_dict(data, config_space)
    elif isinstance(data, dict) and 'observations' in data:
        return History.load_json(history_path, config_space)
    else:
        raise ValueError("Invalid history format. Expected list of observations or History dict.")


# Generate mock history (used when --history not provided)
def generate_mock_history(config_space: ConfigurationSpace, n_samples: int = 50) -> History:
    history = History(task_id='visualization_demo', num_objectives=1, config_space=config_space)
    for _ in range(n_samples):
        config = config_space.sample_configuration()
        config_dict = config.get_dictionary()
        # Simple mock objective
        lr = config_dict.get('learning_rate', 0.01)
        bs = config_dict.get('batch_size', 32)
        hs = config_dict.get('hidden_size', 256)
        y = abs(np.log10(lr) + 3.0) * 15 + abs(bs - 128) * 0.05 + abs(hs - 512) * 0.01
        y = max(y + np.random.normal(0, 3), 0.5)
        obs = Observation(config=config, objectives=[y], trial_state=SUCCESS, elapsed_time=0.1)
        history.update_observation(obs)
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compressor Visualization Example')
    parser.add_argument('--config-space', type=str, default=None,
                        help='Path to config_space.json file')
    parser.add_argument('--history', type=str, default=None,
                        help='Path to history.json file or directory')
    parser.add_argument('--output-dir', type=str, default='./results/visualization_demo',
                        help='Output directory for compression results')
    parser.add_argument('--mode', type=str, default='advanced', choices=['basic', 'advanced'],
                        help='Visualization mode: basic (static HTML) or advanced (local server)')
    parser.add_argument('--port', type=int, default=8050,
                        help='Server port (advanced mode only)')
    args = parser.parse_args()

    # Load or create config space
    if args.config_space:
        print(f"Loading config space from: {args.config_space}")
        config_space = load_config_space(args.config_space)
    else:
        print("Using default config space (mock data)")
        config_space = create_default_config_space()

    # Load or generate history
    if args.history:
        print(f"Loading history from: {args.history}")
        history = load_history(args.history, config_space)
    else:
        print("Generating mock history (50 samples)")
        history = generate_mock_history(config_space, n_samples=50)

    print(f"Config space: {len(config_space.get_hyperparameters())} dimensions")
    print(f"History: {len(history.observations)} observations")

    # Create compressor
    compressor = Compressor(
        config_space=config_space,
        steps=[
            SHAPDimensionStep(strategy='shap', topk=6),
            BoundaryRangeStep(method='boundary', top_ratio=0.8, sigma=2.0)
        ],
        save_compression_info=True,
        output_dir=args.output_dir,
    )

    # Run compression
    surrogate_space, sample_space = compressor.compress_space(space_history=[history])
    print(f"Compression: {len(config_space.get_hyperparameters())} -> {len(surrogate_space.get_hyperparameters())} dims")

    # Visualize results
    if args.mode == 'basic':
        html_path = compressor.visualize_html()
        print(f"HTML saved to: {html_path}")
    else:
        print(f"Starting server on port {args.port}... (Ctrl+C to stop)")
        compressor.visualize_server(port=args.port)
