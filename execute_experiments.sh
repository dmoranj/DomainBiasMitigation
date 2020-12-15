#!/usr/bin/env bash

CIFAR=('cifar-s_domain_discriminative'
       'cifar-s_domain_independent' 'cifar-s_uniconf_adv' 'cifar-s_gradproj_adv'
       'cifar-i_baseline' 'cifar-i_sampling' 'cifar-i_domain_discriminative' 'cifar-i_domain_independent'
       'cifar-c_28_baseline' 'cifar-c_28_sampling' 'cifar-c_28_domain_discriminative' 'cifar-c_28_domain_independent'
       'cifar-d_16_baseline' 'cifar-d_16_sampling' 'cifar-d_16_domain_discriminative' 'cifar-d_16_domain_independent'
       'cifar-d_8_baseline' 'cifar-d_8_sampling' 'cifar-d_8_domain_discriminative' 'cifar-d_8_domain_independent')

CELEBA=('celeba_baseline' 'celeba_weighting' 'celeba_domain_discriminative'
        'celeba_domain_independent' 'celeba_uniconf_adv' 'celeba_gradproj_adv')

RESULTS_DIR=./data/results

if [[ $1 = "cifar" ]]; then
  EXPERIMENTS="${CIFAR[@]}"
elif [[ $1 = "celeba" ]]; then
  EXPERIMENTS="${CELEBA[@]}"
fi

for experiment in $EXPERIMENTS; do
  echo "Executing experiment [$experiment]"
  python3 main.py --experiment $experiment --experiment_name e1 --random_seed 1
done


