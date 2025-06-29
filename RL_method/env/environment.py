from pathlib import Path
import random
import gym
from gym import spaces
import numpy as np


class FileGuessEnv(gym.Env):
    """Gym environment for file content guessing game.

    The agent tries to reconstruct the hexadecimal representation of randomly
    selected files from a directory. Observations and actions are hexadecimal strings.

    Args:
        path2file: Directory path containing files to use as targets
        max_steps: Maximum steps per episode before termination
    """

    def __init__(self, path2file: Path, max_steps: int = 100):
        super().__init__()
        # Collect all file paths from directory (recursive)
        self.list_of_files = [x for x in path2file.glob('**/*') if x.is_file()]
        self.max_steps = max_steps

        # Calculate maximum hex length (2x byte length) for observation/action spaces
        max_file_len = max(len(f.read_bytes()) for f in self.list_of_files) * 2
        # Define hex string observation space (0-9, a-f)
        self.observation_space = spaces.Text(max_length=max_file_len, charset="0123456789abcdef")
        # Define action space (hex string predictions)
        self.action_space = spaces.Text(max_length=max_file_len, charset="0123456789abcdef")

        # Environment state tracking
        self.current_file = None  # Currently selected file path
        self.current_data = None  # Current target hex string
        self.step_count = 0  # Steps in current episode

    def reset(self):
        """Reset environment state and start new episode.

        Returns:
            observation (str): Hexadecimal representation of randomly selected file
        """
        self.current_file = random.choice(self.list_of_files)
        self.current_data = self._read_file(self.current_file)
        self.step_count = 0
        return self.current_data

    def step(self, action: str):
        """Execute agent action and transition to next state.

        Args:
            action (str): Agent's predicted hex string

        Returns:
            tuple: (next_observation, reward, done, info)
                next_observation: Hex string of next state (same file by default)
                reward: Similarity score between action and target
                done: Episode termination flag
                info: Metadata containing current file path
        """
        self.step_count += 1

        # Ensure action is string (handle possible sequence inputs)
        if not isinstance(action, str):
            action = "".join(action)

        # Calculate reward based on hex string similarity
        reward = self._calculate_reward(self.current_data, action)

        # Check episode termination conditions
        done = self._is_done(reward)

        # Prepare next state
        next_data = None
        if not done:
            # Keep same file for next step (Option 1)
            next_data = self.current_data
            # Alternative: Random new file (Option 2 - uncomment to use)
            # self.current_file = random.choice(self.list_of_files)
            # next_data = self._read_file(self.current_file)
        else:
            next_data = self.current_data  # Maintain terminal state consistency

        return next_data, reward, done, {"file": str(self.current_file)}

    def _read_file(self, file_path: Path) -> str:
        """Read file and convert to hexadecimal string representation.

        Args:
            file_path: Path object of target file

        Returns:
            str: Hexadecimal string of file contents
        """
        with open(file_path, 'rb') as f:
            return f.read().hex()

    def _calculate_reward(self, target: str, prediction: str) -> float:
        """Calculate normalized similarity reward between target and prediction.

        Handles variable-length strings using:
        reward = matching_chars / max_length

        Args:
            target: Ground truth hex string
            prediction: Agent-generated hex string

        Returns:
            float: Reward value [0.0, 1.0]
        """
        if len(target) != len(prediction):
            min_len = min(len(target), len(prediction))
            correct = sum(1 for i in range(min_len) if target[i] == prediction[i])
            return correct / max(len(target), len(prediction))
        return sum(1 for t, p in zip(target, prediction) if t == p) / len(target)

    def _is_done(self, reward: float) -> bool:
        """Determine episode termination conditions.

        Terminate when:
        - Maximum steps reached
        - Agent achieves ≥80% accuracy

        Args:
            reward: Current similarity score

        Returns:
            bool: Termination signal
        """
        return self.step_count >= self.max_steps or reward >= 0.8

    def _select_next_file(self):
        """Select next file for environment state (current implementation uses same file).

        Note: Alternate implementation (random new file) is commented in step()

        Returns:
            str: Hexadecimal content of current file
        """
        # Maintains current file across steps
        return self.current_data