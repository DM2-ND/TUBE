# TUBE: Embedding Behavior Outcomes for Predicting Success

**Description: This repository contains the C++ implementation of the TUBE model proposed in paper [TUBE: Embedding Behavior Outcomes for Predicting Success](http://www.meng-jiang.com/pubs/tube-kdd19/tube-kdd19-paper.pdf) accepted by KDD 2019**

## Usage
### 1. Make
To make the executable file, please change into the project folder and run:
```
make all
```
**Notes:**
This program uses `gnu++11` standard. If you need advanced control over options when compiling the program, please look into the `./Makefile` file.

### 2. Execute
After the executable file `./tube` is generated, run:
```
./tube --input_behaviors_file data/synthetic-skill03.txt --output_goal_embs_file goals.txt --output_context_embs_file contexts.txt --dims 16 --negative 2 --threads 4 --samples 1
```
List of parameters:
+ --input_behaviors_file: The input file of training behaviors. Each line follows format `<goal>\t<context_1>[,<context_2>,...]`
+ --output_goal_embs_file: The output file of goal embeddings. First line is header: `<#goal>\t<#dimension>`. Then, each line follows format `<goal>\t<dimension_1>\t<dimension_2>\t...\t<dimension_n>`
+ --output_context_embs_file: The output file of context embeddings. First line is header: `<#context>\t<#dimension>`. Then, each line follows format `<context>\t<dimension_1>\t<dimension_2>\t...\t<dimension_n>`
+ --dims: The number of dimensions of the embedding; default is 8.
+ --threads: The number of threads used for training; default is 2.
+ --samples: The number of behavior samples used for traninig in **millions**; default value equals to the number of behaviors in *--input_behaviors_file* X **500**.
+ --negative: The number of negative samples; default is 1.
+ --rate: The initial value of learning rate; default is 0.005.

## Data
A pre-processed demo dataset is included.
+ `./data/synthetic-skill03.txt`: The synthetic behavior dataset with `k=3` as described in the paper.

Other datasets can be found at:
+ Complete synthetic datasets: <https://bit.ly/2Uyx2Io>
+ Real publication dataset with 2k venues: <https://bit.ly/2GgbqfP>
+ Real publication dataset with 5k venues: <https://bit.ly/2TtJatW>
+ Real publication dataset with 10k venues: <https://bit.ly/2UuOfSI>

## Example
Other examples are provided in the `./demo.sh` file.

## Miscellaneous
If you find this code package to be useful, please consider cite us:
```
@inproceedings{wang2019tube,
  title={Tube: Embedding behavior outcomes for predicting success},
  author={Wang, Daheng and Jiang, Tianwen and Chawla, Nitesh V and Jiang, Meng},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1682--1690},
  year={2019}
}
```
