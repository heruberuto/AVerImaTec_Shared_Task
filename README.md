# AVerImaTeC Shared Task

# 🎊 News <!-- omit in toc -->

- [2024.09] 🔥 AVerImaTeC shared task has been held [here](https://huggingface.co/spaces/FEVER-IT/AVerImaTeC). 
- [2025.09] 🎉 We are pleased to announce that FEVER9 will be co-located with EACL2026! In this year's workshop, we will focus on image-text claim verification and leverage AVerImaTeC as the shared task. You can learn more about FEVER9 and past FEVER workshops [here](https://fever.ai/index.html).
- [2025.09] 🎉 Our AVerImaTeC paper is accepted by NeurIPS Datasets and Benchmarks track! You can access the lastest version of the paper at [here](https://arxiv.org/pdf/2505.17978).

# Baseline Implementation for AVerImaTeC Shared Task

This repository maintains the baseline for [AVerImaTeC shared task](https://huggingface.co/spaces/FEVER-IT/AVerImaTeC). The shared task is built on top of the dataset, AVERIMATEC: A Dataset for Automatic Verification of Image-Text Claims with Evidence from the Web. You can find the paper [here](https://arxiv.org/pdf/2505.17978). To reduce the expense of web search, we provide the knowledge store containing the right eivdence for claim verification as well as noisy evidence in this [link](https://drive.google.com/drive/folders/1vjy7mjA4NTuLQfPh5-NZFpaxn8_H9rUs?usp=sharing). The code implementation is compatible with using the provided knowledge store.

## Content
- [Dataset Preparation](#dataset-preparation)
- [Experiment Setting](#experiment-setting)
- [Baseline Implementation](#baseline-implementation)
- [Baseline Evaluation](#baseline-evaluation)

## Dataset Preparation

### AVerImaTeC Data
Please download the data from our provided [link](https://huggingface.co/datasets/Rui4416/AVerImaTeC). Put the *images.zip* under the *data/data_clean* folder and unzip it. For json files, please put it under the *data/data_clean/split_data*. 

### Evidence Source
In order to use our provided knowledge store, please download the files from [here](https://drive.google.com/drive/folders/1vjy7mjA4NTuLQfPh5-NZFpaxn8_H9rUs?usp=sharing). 
If you would like to do your own web search, in order to use Google Search, you need to put your own API keys in the folder *private_info*.

## Experiment Setting
In order to implement our baselines, you need to install essential packages listed in *requirement.txt*.

## Baseline Implementation
In order to generate predictions with our baselines, please refer to the [provided script](https://github.com/abril4416/AVerImaTec_Shared_Task/blob/main/scripts/baseline_script.sh), and make sure you set the *[ROOT_PATH]* and *[DATASTORE_PATH]* correctly. You can also set *--DEBUG True* to switch to the debug mode (only test a few claims) for easy debugging.
We provide the implementation for the following LLMs and MLLMs: LLM_NAME can be set to *gemini-2.0-flash-001*, *qwen* and *gemma* while MLLM_NAM can be set to *gemini-2.0-flash-001*, *qwen*, *gemma* and *llava*.

Different strategies of question generation (described in Section 7.1 in the [paper](https://arxiv.org/pdf/2505.17978)) are available in the implementation. You can change *--HYBRID_QG*, *--PARA_QG* to be True to adopt the hybrid and parallel question generation, otherwise, dynamic question generation. By setting *--QG_ICL* to be True, you are using few-shot in-context learning for question generation.

If you would like to submit your predictions to our [leaderboard](https://huggingface.co/spaces/FEVER-IT/AVerImaTeC), please use our [code](https://github.com/abril4416/AVerImaTec_Shared_Task/blob/main/prepare_submission/ipython/Result_Convert.ipynb) to covert the baseline outputs to the required format of submission.


## Baseline Evaluation
After post-processing to generate the submission file with the [code](https://github.com/abril4416/AVerImaTec_Shared_Task/blob/main/prepare_submission/ipython/Result_Convert.ipynb), you can use the [script](https://github.com/abril4416/AVerImaTec_Shared_Task/blob/main/scripts/eval_offline.sh) to do your own evaluation. After runnimg the script, you can view the generated scores with the code [here](https://github.com/abril4416/AVerImaTec_Shared_Task/blob/main/prepare_submission/ipython/Eval_Score_Compute.ipynb). It returns the question score, evidence score, conditional verdict accuracy and conditional justification score. You can set *threshold* to different values to set different thresholds of the evidence scores (more details in Section6 in the [paper](https://arxiv.org/pdf/2505.17978)).
