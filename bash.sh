#!/bin/bash

for i in {1..1}
do
	python get_data.py --trajectories 100
	python train_model.py
	python train_model.py
  python test_models.py
done
