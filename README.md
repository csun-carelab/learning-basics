# INSTRUCTIONS

## Installing

- You will need to install [Python](https://www.python.org/downloads/) and [Pytorch](https://pytorch.org/).
- See [these tutorials](https://docs.pytorch.org/tutorials/) for how to use Pytorch.

## Running the code

The files should be run in the following order:
1. Collect a dataset of robot trajectories
   ```python get_data.py```
2. Train a neural network policy
   ```python train_model.py```
3. Test your trained robot
   ```python test_model.py```

You may do this by directly running the bash script
```./bash.sh```
