# SpaceShooter
A Basic space shooter environment simulated using pygame and trained with a chosen RL policy.

- **Conda Environment**: Skip the first two lines if you have already installed conda.

```bash
# Download anaxonda3
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh

# Install anaconda3 (follow the on-screen instructions)
bash ./Anaconda3-2024.02-1-Linux-x86_64.sh

# Create new environment named 'mlops' with necessary specifications
conda create -n mlops python=3.8
conda activate mlops
conda install pytorch torchvision torchaudio cudatoolkit
conda install -c conda-forge gym=0.21.0 opencv=4.5.5
pip install stable-baselines3==1.5.0 
pip install pygame
```
- **Utilizing the YAML file**: Install using the .yaml file to preserve dependencies

```bash
# Modify the yaml file last line prefix to your anaconda or miniconda installation directory - and then continue the above steps
conda env create -f environment_linux.yaml
```
<br>

- **Training**: Code is given below to train the agent.

```bash
# Start training multi agent with separate scoring mechanism for each agent
python3 runner.py

# For training multi agent with fleet mechanism, change the line 5 in runner.py to
from env import MultiAgentSpaceShooterEnv
```
