import yaml
from pathlib import Path

def load_config(config_path: str | Path) -> dict:
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_config(config_name: str, config_dir: str | Path = "config") -> dict:
    """Gets a specific configuration by name."""
    config_dir = Path(config_dir)
    config_file = config_dir / f"{config_name}.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    return load_config(config_file)

if __name__ == '__main__':
    # Example usage:
    try:
        model_config = get_config("model_config")
        print("Model Config:", model_config)
        
        prompt_config = get_config("prompt_templates")
        print("Prompt Config:", prompt_config)
    except FileNotFoundError as e:
        print(e) 