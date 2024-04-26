#!/bin/bash


for shot in 1 2 4 8 50; do for seed in 1 2 3 4 5; do for alpha in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do python multimodal_prompt_fewshot_lxr.py --shot $shot --seed $seed --alpha $alpha --full false --data "goss"; done; done; done

for shot in 1 2 4 8 25; do for seed in 1 2 3 4 5; do for alpha in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do python multimodal_prompt_fewshot_lxr.py --shot $shot --seed $seed --alpha $alpha --full false --data "poli"; done; done; done


for seed in 1; do for alpha in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do python multimodal_prompt_fewshot_lxr.py --seed $seed --alpha $alpha --full true --data "goss"; done; done


for seed in 1 2 3 4 5; do for alpha in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do python multimodal_prompt_fewshot_lxr.py --seed $seed --alpha $alpha --full true --data "goss"; done; done