# Central configuration for Weights & Biases integration.

# Choose one of the following modes:
# 'online': (Default) Syncs data to the cloud in real-time. Requires a stable internet connection.
# 'offline': Saves data locally. Use this if you have network issues. Sync later with `wandb sync`.
# 'disabled': Completely disables W&B integration.
WANDB_MODE = "offline" 