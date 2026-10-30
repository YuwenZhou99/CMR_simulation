# ur-simulation

Repository for control methods of a simulated UR7e

## Installation

```bash
git clone https://github.com/YuwenZhou99/CMR_simulation.git
pip install -e control_methods_simulation/
pip install -r requirements.txt
```

## Usage

### Model-based control for free-space motion

Point-to-point trajectory:
```bash
python scripts/classic_control.py --mode point
```

Path trajectory:
```bash
python scripts/classic_control.py --mode path
```

### Model-based control for object pushing

```bash
python scripts/classic_control_env.py
```

### RL control for free-space motion

Training process:
```bash
python scripts/rl_control.py train --task trajectory --mode point --steps 1000000 --seed 0 --output results/rl/point_ppo
python scripts/rl_control.py train --task trajectory --mode path --steps 1000000 --seed 0 --output results/rl/path_ppo
```

Evaluate the results:
```bash
python scripts/rl_control.py evaluate --checkpoint results/rl/point_ppo --episodes 1 --gui --output results/rl/point_ppo/evaluation
python scripts/rl_control.py evaluate --checkpoint results/rl/path_ppo --episodes 1 --gui --output results/rl/path_ppo/evaluation
```

### RL control for object pushing

Train in the fix environment:
```bash
python scripts/rl_control.py train --task push --steps 1000000 --seed 0 --output results/rl/push_ppo
python scripts/rl_control.py evaluate --checkpoint results/rl/push_ppo --scenes fixed --episodes 1 --seed 0 --output results/rl/push_ppo/evaluation_fixed --gui
python scripts/rl_control.py evaluate --checkpoint results/rl/push_ppo --scenes random --episodes 20 --seed 30000 --output results/rl/push_ppo/evaluation_random
```

Continue training with randomized scenarios starting from the model:
```bash
python scripts/rl_control.py train --resume results/rl/push_ppo --random-positions --steps 200000 --seed 0 --output results/rl/push_random_finetune
python scripts/rl_control.py evaluate --checkpoint results/rl/push_random_finetune --scenes fixed --episodes 1 --seed 0 --output results/rl/push_random_finetune/evaluation_fixed --gui
python scripts/rl_control.py evaluate --checkpoint results/rl/push_random_finetune --scenes random --episodes 20 --seed 30000 --output results/rl/push_random_finetune/evaluation_random
```

Evaluation saves data and plots, then opens the plots automatically. Add `--record` to record an evaluation video; recording automatically enables the GUI.

