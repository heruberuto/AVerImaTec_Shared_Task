import json
import os
import pickle as pkl
import random
import argparse

import sys

sys.path.append("..")
root_dir = os.path.abspath("..")
from ref_eval import val_evid_idv, compute_image_scores, textual_val_single
from qa_to_evidence import qa_to_evid
from utils import convert_qa_format
import utils

import torch

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def load_pkl(path):
    data = pkl.load(open(path, "rb"))
    return data


def load_json(path):
    data = json.load(open(path, "r"))
    return data


import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate extra questions based on claims with a prompt. Useful for searching."
    )
    parser.add_argument("--eval_model", default="google/gemma-3-27b-it")
    parser.add_argument("--llm_name", default="gemma")
    parser.add_argument("--mllm_name", default="gemma")
    parser.add_argument(
        "--pred_file_path", default="/mnt/data/factcheck/averimatec/submissions/submission.json"
    )
    parser.add_argument(
        "--root_dir", default="/mnt/data/factcheck/averimatec"
    )  # this is the absolute path where you put AVerImaTec.
    parser.add_argument(
        "--cache_dir", default=""
    )  # this is the absolute path where you save your huggingface model
    parser.add_argument("--save_num", type=str, default="4")
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()

    """
    Potential issues related to Gemma: https://github.com/google-deepmind/gemma/issues/169
    """
    mllm_name = args.eval_model
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    model = Gemma3ForConditionalGeneration.from_pretrained(
        mllm_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
        # attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(mllm_name, cache_dir=args.cache_dir)
    mllm = {"model": model.eval(), "processor": processor}

    print("Root dir:", args.root_dir)
    #p2_data = load_json(os.path.join(args.root_dir, "data/data_clean/split_data/test.json"))
    p2_data = load_json(os.path.join(args.root_dir, "val.json"))
    if len(args.pred_file_path) > 0:
        pred_file = load_json(args.pred_file_path)
    else:
        pred_file = load_json(
            os.path.join(
                args.root_dir,
                "prepare_submission/converted_results",
                "_".join([args.llm_name, args.mllm_name, str(args.save_num)]) + ".json",
            )
        )

    if os.path.exists(os.path.join(args.root_dir, "prepare_submission/converted_results", "gt_evid.json")):
        gt_evid_flag = True
        gt_evid_set = load_json(
            os.path.join(args.root_dir, "prepare_submission/converted_results", "gt_evid.json")
        )
    else:
        gt_evid_flag = False
        gt_evid_set = []

    if os.path.exists(
        os.path.join(
            args.root_dir,
            "prepare_submission/intermediate_eval_results",
            "_".join([args.llm_name, args.mllm_name, str(args.save_num)]) + ".json",
        )
    ):
        all_eval_results = load_json(
            os.path.join(
                args.root_dir,
                "prepare_submission/intermediate_eval_results",
                "_".join([args.llm_name, args.mllm_name, str(args.save_num)]) + ".json",
            )
        )
    else:
        all_eval_results = []
    for i, row in enumerate(p2_data):
        if args.debug and i > 4:
            break
        if i % 20 == 0:
            print(i, "saving...")
            json.dump(
                all_eval_results,
                open(
                    os.path.join(
                        args.root_dir,
                        "prepare_submission/intermediate_eval_results",
                        "_".join([args.llm_name, args.mllm_name, str(args.save_num)]) + ".json",
                    ),
                    "w",
                ),
            )
        req_id = i
        gt_questions = [info["question"] for info in row["questions"]]
        gt_justification = row["justification"]
        gt_verdict = row["label"]
        if gt_evid_flag:
            gt_evid = gt_evid_set[i]
        else:
            gt_evid = [convert_qa_format(qa, mllm, mllm_name, args.root_dir) for qa in row["questions"]]
            gt_evid_set.append(gt_evid)

        pred_evid = pred_file[i]["evidence"]
        pred_justi = pred_file[i]["justification"]
        pred_questions = pred_file[i]["questions"]
        pred_verdict = pred_file[i]["verdict"]

        # verdict prediction
        if pred_verdict.lower().strip() == gt_verdict.lower():
            verdict_acc = 1.0
        else:
            verdict_acc = 0.0
        # evidence evaluation
        detailed_evid_val, evid_val_score = val_evid_idv(mllm, mllm_name, pred_evid, gt_evid, False, True)
        img_scores = compute_image_scores(mllm, mllm_name, pred_evid, gt_evid, evid_val_score)
        _, evid_acc, _ = utils.get_auto_recall(detailed_evid_val, img_scores, len(gt_evid), len(pred_evid))
        # justification generation
        justi_feedback, justi_score = textual_val_single(
            gt_justification, pred_justi, args.root_dir, mllm_name, mllm, "justification", args.debug
        )
        justi_acc = utils.justi_recall_compute(justi_feedback, justi_score)
        # question generation
        ques_feedback, ques_score = textual_val_single(
            gt_questions, pred_questions, args.root_dir, mllm_name, mllm, "question", args.debug
        )
        ques_acc = utils.ques_recall_compute(ques_score, len(gt_questions), len(pred_questions))
        if args.debug:
            print("##Question:\n", ques_feedback, "\n", ques_score, "\n\t", ques_acc)
            print("##Verdict:\n", pred_verdict, gt_verdict, verdict_acc)
            print("##Evidence:\n", detailed_evid_val, "\n", img_scores, "\n\t", evid_acc)
            print("##Justification:\n", justi_feedback, "\n", justi_score, "\n\t", justi_acc)

        all_eval_results.append(
            {
                "ques_score": ques_acc,
                "evid_score": evid_acc,
                "verdict_score": verdict_acc,
                "justi_score": justi_acc,
                "intermediate_info": {
                    "ques_feedback": ques_feedback,
                    "ques_score": ques_score,
                    "justi_feedback": justi_feedback,
                    "justi_score": justi_score,
                    "evid_feedback": detailed_evid_val,
                    "evid_image_score": img_scores,
                    "evid_text_score": evid_val_score,
                },
            }
        )

    json.dump(
        all_eval_results,
        open(
            os.path.join(
                args.root_dir,
                "prepare_submission/intermediate_eval_results",
                "_".join([args.llm_name, args.mllm_name, str(args.save_num)]) + ".json",
            ),
            "w",
        ),
    )
    if gt_evid_flag == False:
        json.dump(
            gt_evid_set,
            open(os.path.join(args.root_dir, "prepare_submission/converted_results", "gt_evid.json"), "w"),
        )
