import re
import os

threshold = 9
from qa_to_evidence import qa_to_evid
from collections import defaultdict


def compute_scores(score, len_gt_evid, len_val_evid):
    if len_val_evid == 0 or len_gt_evid == 0:
        return 0.0, 0.0, 0.0
    precision = score["pred_in_ref"] / len_val_evid
    recall = score["ref_in_pred"] / len_gt_evid
    if precision < 0:
        precision = 0
    if recall < 0:
        recall = 0
    if recall > 1.0:
        recall = 1.0
    if precision > 1.0:
        precision = 1.0
    if precision == 0 and recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def justi_recall_compute(raw_gen, score):
    pred_facts = raw_gen.strip().split("[PRED in REF Exp]: ")[1].split("[REF in PRED]:")[0].strip()
    ref_facts = raw_gen.strip().split("[REF in PRED Exp]:")[-1].strip()

    num_gt = len(re.findall(r"\b\d+\.\s", ref_facts))
    num_pred = len(re.findall(r"\b\d+\.\s", pred_facts))

    _, recall, _ = compute_scores(score, num_gt, num_pred)
    return recall


def ques_recall_compute(score, num_gt, num_pred):
    _, recall, _ = compute_scores(score, num_gt, num_pred)
    return recall


def compute_scores_detail(score, len_gt_evid, len_val_evid):
    if len_val_evid == 0 or len_gt_evid == 0:
        return None, None, None
    # print (score["pred_in_ref"])
    precision = score["pred_in_ref"] / len_val_evid
    recall = score["ref_in_pred"] / len_gt_evid
    if precision < 0:
        precision = 0
    if recall < 0:
        recall = 0
    if recall > 1.0:
        recall = 1.0
    if precision > 1.0:
        precision = 1.0
    if precision == 0 and recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def get_auto_recall(result, image_scores, len_ref, len_pred):
    # print (result)
    pred_in_ref = image_scores["pred_in_ref"]
    ref_in_pred = image_scores["ref_in_pred"]
    # print (pred_in_ref)
    pred_dict = defaultdict(int)
    # len_pred=len(pred_evid)
    # print (ref_in_pred)
    num_pred_in_ref = 0
    for i, info in enumerate(pred_in_ref):
        # print (info)
        try:
            pred_idx = int(info["info"][0].split("_")[-1])
            ref_idx = int(info["info"][1].split("_")[-1])
        except:
            continue
        if pred_idx in pred_dict:
            continue
        pred_dict[pred_idx] += 1
        try:
            if int(info["score"]) < threshold:
                continue
            else:
                num_pred_in_ref += 1
        except:
            continue
    ref_dict = defaultdict(int)
    num_ref_in_pred = 0
    for i, info in enumerate(ref_in_pred):
        # print (info)
        try:
            pred_idx = int(info["info"][1].split("_")[-1])
            ref_idx = int(info["info"][0].split("_")[-1])
        except:
            continue
        if ref_idx in ref_dict:
            continue
        ref_dict[ref_idx] += 1
        # print (ref_dict)
        try:
            if int(info["score"]) < threshold:
                continue
            else:
                num_ref_in_pred += 1
        except:
            continue
    """
    print (result['detailed_val'])
    print ('\t',pred_in_ref)
    print ('\t',ref_in_pred)
    print ('\t',num_pred_in_ref,num_ref_in_pred)
    """
    precision, recall, f1 = compute_scores_detail(
        {"ref_in_pred": num_ref_in_pred, "pred_in_ref": num_pred_in_ref}, len_ref, len_pred
    )
    return precision, recall, f1


def convert_qa_format(question_info, llm, llm_name, root_dir):
    answers = question_info["answers"]
    ques_txt = question_info["question"].replace("\n", "; ")
    related_images = []
    ques_img_str = []
    ans_text = []
    if len(question_info["input_images"]):
        rel_images = question_info["input_images"]
        for image in rel_images:
            related_images.append(os.path.join(root_dir, "data/data_clean/images", image))
            ques_img_str.append("[IMG_" + str(len(related_images)) + "]")
    for j, answer in enumerate(answers):
        answer_type = answer["answer_type"]
        if answer_type == "Image":
            image_answers = answer["image_answers"]
            for image in image_answers:

                related_images.append(os.path.join(root_dir, "data/data_clean/images", image))
                ans_text.append("[IMG_" + str(len(related_images)) + "]")
        else:
            ans_text.append(answer["answer_text"])
        if answer_type == "Boolean":
            boolean_explanation = answer["boolean_explanation"]
            ans_text.append(boolean_explanation)
    ans_text = " ".join(ans_text).replace("\n", "; ")
    if len(ques_img_str):
        evid_ques = ques_txt + ", ".join(ques_img_str)
    else:
        evid_ques = ques_txt
    evid = qa_to_evid(evid_ques, ans_text, llm, llm_name)
    # print (evid)
    # print (related_images)
    evid_info = {"text": evid, "images": related_images}
    return evid_info
