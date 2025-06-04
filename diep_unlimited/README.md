## Diep.io - Unlimited Observations

This directory contains code for the **Unlimited Observation** part of the report.

It uses deep sets to process observations, and the agent itself is based on SAC.

The requirements are the same as the rest of the project.

### Usage
- To train an agent:
```
python teacher.py
```

- To watch a trained agent play:
```
python env.py
```

### Pre-trained agent
A pre-trained agent file is included in `saves/`. It's not very good.
The main function in `env.py` should specify the path of the models used.