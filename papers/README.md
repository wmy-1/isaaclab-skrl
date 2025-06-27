# 1.DeXtreme: Transfer of Agile In-hand Manipulation from Simulation to Reality

1. training
```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 scripts/reinforcement_learning/skrl/train.py --task Isaac-Repose-Cube-Allegro-v0 --distributed --headless
```
```bash
python scripts/reinforcement_learning/skrl/play.py --task Isaac-Repose-Cube-Allegro-v0 --headless --video
```
2. dive into paper
💡 There is a challenging gap for multi-finger manipulation model transfering from simulation to the real world.

2. dive into code
