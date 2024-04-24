#!/bin/bash

#shot2=(2 4 8)
#shot=(1 2 4 8 50)
#seed=(1 2 3 4 5)
#alpha=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
#alpha2=(0.9 1.0)

#for seed in "${seed[@]}"; do
#    for alpha in "${alpha[@]}"; do
#        python multimodal_prompt_fewshot.py --shot 50 --seed "$seed" --alpha "$alpha" --full false --data "goss"
#    done
#done


# few
#for shot in "${shot[@]}"; do
#    for seed in "${seed[@]}"; do
#        for alpha in "${alpha[@]}"; do
#	    python multimodal_prompt_fewshot.py --shot "$shot" --seed "$seed" --alpha "$alpha" --full false --data "poli"
#	done
#    done
#done

# full
#for alpha in "${alpha[@]}"; do
#    python multimodal_prompt_fewshot.py --shot 0 --seed 0 --alpha "$alpha" --full true --data "poli"
#done


#for seed in "${seed[@]}";do
#    for alpha in "${alpha[@]}";do
#        python multimodal_prompt_fewshot.py --shot 50 --seed "$seed" --alpha 0.0 --full false --data "poli"
#    done
#done

#for shot in "${shot[@]}"; do
#    for alpha in "${alpha[@]}";do
#        python multimodal_prompt_fewshot.py --shot "$shot" --seed 1 --alpha "$alpha" --full false --data "poli" 
#    done
#done

#for shot2 in "${shot2[@]}";do
#    python multimodal_prompt_fewshot.py --shot "$shot2" --seed 1 --alpha 0.0 --full false --data "poli"
#done

#

# for shot in 1; do for seed in 1; do for alpha in 1; do python multimodal_prompt_fewshot_lxr.py --shot $shot --seed $seed --alpha $alpha --full false --data "goss"; done; done; done

# for shot in 1; do for seed in 2; do for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do python multimodal_prompt_fewshot_lxr.py --shot $shot --seed $seed --alpha $alpha --full false --data "goss"; done; done; done

# for shot in 1 2 4 8 25; do for seed in 1 2 3 4 5; do for alpha in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do python multimodal_prompt_fewshot_lxr.py --shot $shot --seed $seed --alpha $alpha --full true --data "poli"; done; done; done
# for seed in 1; do for alpha in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do python multimodal_prompt_fewshot_lxr.py --seed $seed --alpha $alpha --full true --data "goss"; done; done



# for seed in 2; do for alpha in 0.7 0.8 0.9 1; do python multimodal_prompt_fewshot_lxr.py --seed $seed --alpha $alpha --full false --data "poli"; done; done
#
# for shot in 50; do for seed in 1 2 3 4 5; do for alpha in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do python multimodal_prompt_fewshot_lxr.py --shot $shot --seed $seed --alpha $alpha --full false --data "goss"; done; done; done

# for seed in 1; do for alpha in 0.4; do python multimodal_prompt_fewshot_lxr.py --seed $seed --alpha $alpha --full true --data "goss"; done; done
# for shot in 8; do for seed in 2; do for alpha in 1; do python multimodal_prompt_fewshot_lxr.py --shot $shot --seed $seed --alpha $alpha --full false --data "goss"; done; done; done

# for shot in 8; do for seed in 3 4 5; do for alpha in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do python multimodal_prompt_fewshot_lxr.py --shot $shot --seed $seed --alpha $alpha --full false --data "goss"; done; done; done

for shot in 1 2 4 8 50; do for seed in 1 2 3 4 5; do for alpha in 0.5; do python multimodal_prompt_fewshot_lxr.py --shot $shot --seed $seed --alpha $alpha --full false --data "goss"; done; done; done
#
#for seed in 1; do for alpha in 0 0.1 0.2 0.3 0.4 0.5; do python multimodal_prompt_fewshot_lxr.py --seed $seed --alpha $alpha --full true --data "goss"; done; done

#python multimodal_prompt_fewshot_lxr.py --shot 1 --seed 1 --alpha 1 --full false --data "poli"

