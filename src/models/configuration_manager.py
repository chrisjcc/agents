import yaml
from schema import Schema, SchemaError

# Define the schema for the YAML file
config_schema = Schema(
    {
        "env_name": str,
        "max_episode_steps": int,
        "num_episodes": int,
        "domain_randomize": bool,
        "continuous": bool,
        "render_mode": str,
        "use_gae": bool,
        "gamma": float,
        "lr": float,
        "value_coef": float,
        "entropy_coef": float,
        "hidden_dim": int,
        "batch_size": int,
        "checkpoint_dir": str,
        "checkpoint_freq": int,
        "num_checkpoints": int,
    }
)


class ConfigurationManager:
    def __init__(self, config_file):
        # Load the YAML file
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        # Validate the YAML file against the schema
        try:
            config_schema.validate(config)
        except SchemaError as e:
            print(f"Invalid configuration file: {e}")

        self.env_name = config["env_name"]
        self.max_episode_steps = config.get("max_episode_steps", 600)
        self.num_episodes = config.get("num_episodes", 10)
        self.domain_randomize = config.get("domain_randomize", True)
        self.continuous = config.get("continuous", True)
        self.render_mode = config.get("render_mode", "human")
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("lr", 0.0001)
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.batch_size = config.get("batch_size", 64)
        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        self.checkpoint_freq = config.get("checkpoint_freq", 1)
        self.num_checkpoints = config.get("num_checkpoints", 5)

    def __repr__(self):
        return f"ConfigurationManager(env_name={self.env_name}, max_episode_steps={self.max_episode_steps}, num_episodes={self.num_episodes}, ...)"
