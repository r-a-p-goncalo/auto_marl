

from huggingface_sb3 import load_from_hub


checkpoint = load_from_hub(
	repo_id="sb3/dqn-MountainCar-v0",
	filename="MountainCar-v0.zip",
)