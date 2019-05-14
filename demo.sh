#!/bin/bash

make clean
make all

job_cmd="./tube --input_behaviors_file data/synthetic-skill03.txt --output_goal_embs_file goals.txt --output_context_embs_file contexts.txt --dims 16 --negative 2 --threads 4 --samples 1"

eval $job_cmd
