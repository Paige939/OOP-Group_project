# Project structure
group_project/
│
├── Agents/
│   ├── __init__.py
│   ├── random_agent.py
│   └── ddpg_agent.py
│
├── base_agent.py          
├── env_wrapper.py
├── experiment.py
└── main.py
You can add any other _agent.py into Agents folder that inherit act() and reset() functions in base_agent.py

# Installation
```
pip install gymnasium[classic-control]
pip install pygame
```

# Introduction to Pendulum
