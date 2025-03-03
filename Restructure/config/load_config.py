def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default.
    
    Returns:
        Configuration dictionary
    """
    import yaml
    import os
    
    if config_path is None:
        # Use default configuration
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config