import os
import pickle as pkl
import torch
from PIL import Image
import sys

sys.path.append("..")
import re
import random

root_dir = os.path.abspath(".")
text_val_demo = open(os.path.join(root_dir, "templates/evid_evaluation_text.txt")).readlines()
text_val_demo = "".join(text_val_demo)

seperate_val_demo = open(os.path.join(root_dir, "templates/evid_evaluation_text_seperate.txt")).readlines()
seperate_val_demo = "".join(seperate_val_demo)


joint_val_demo = "".join(
    [
        "You will get as input a reference evidence ([REF]) and a predicted evidence ([PRED]). [IMG_1], [IMG_2] ... are placeholders for images. Two facts may be textually aligned. But if they have different images, the two facts could be irrelevant.\nPlease verify the correctness of the predicted evidence by comparing it to the reference evidence, following these steps:\n1. Evaluate each fact in the predicted evidence individually: is the fact supported by the REFERENCE evidence? Do not use additional sources or background knowledge.\n2. Evaluate each fact in the reference evidence individually: is the fact supported by the PREDICTED evidence? Do not use additional sources or background knowledge.\n3. Finally summarise (1.) how many predicted facts are supported by the reference evidence and explanations([PRED in REF] and [PRED in REF Exp]), (2.) how many reference facts are supported by the predicted evidence and explanations ([REF in PRED] and [REF in PRED Exp]).\nGenerate the output as shown in the examples below:\n","[PRED]: 1. The missle in [IMG_1] is Fateh 110. 2. Ilan Omar has attended the training in [IMG_2]. 3. Prince Phillip wore the Royal Guard uniform in Jan. 14, 2003. 4.The raid in Washington took place on Saturday, Oct. 26, 1999.\n[REF]: 1. [IMG_1] was taken in Jan. 20, 2003. 2. No evidence can be found related to the type of missle in [IMG_2]. 3. The woman in [IMG_3] for a training is not Ilan Omar. 4. No answer was found regarding when the raid in Washington took place. 5. Prince Phillip wore the Royal Guard uniform shown in [IMG_4] previously in Jan. 2003.\n[PRED in REF]: 1\n[PRED in REF Exp]: 1. It is refuted by the second fact in the predicted evidence set. 2. No related facts can be found in the reference set. Though the text part of the thrid fact in the reference set is similar to it, they have different images. 3. The fact is supported by the fifth fact in the evidence set. 4. It is refuted by the fourth fact in the reference evidence set.\n[REF in PRED]: 0\n[REF in PRED Exp]: 1. No related facts can be found in the reference set. Though the textual part of the fact aligns with the third fact in the predicted evidence, they have different images. 2. It is refuted by the first fact in the predicted evidence set. 3. No related facts can be found in the reference set. Though the text part of the second fact in the predicted evidence set is similar to it, they have different images. 4. It is refuted by the fourth fact in the predicted evidence which claims the date of the raid could be found. 5.  No related facts can be found in the reference set.  Though the text part of the third fact in the predicted evidence set is similar to it, they have different images.\n","[PRED]: 1. [IMG_1] was taken on Jan. 19, 2025. 2. The current view of the benches in [IMG_2] is [IMG_3]. 3. The date of the claim is Nov. 22, 2023.\n[REF]: 1. The claim was made on Jan. 22, 2021. 2. [IMG_2] was taken on Jan. 19, 2025. 3. The benches in [IMG_1] currently look like [IMG_3]. 4.Trump dressed as [IMG_1] in the meeting.\n[PRED in REF]: 2\n[PRED in REF Exp]: 1. The second piece of evidence in the reference evidence supports it. 2. The third evidence in the evidence set has a similar meaning to this fact. 3. The fact claims the date as Nov. 22, 2023, which is different from the first fact in the refence evidence, Jan. 22, 2021.\n[REF in PRED]: 2\n[REF in PRED Exp]: 1. The fact claims the date as Jan. 22, 2021, which is different from the third fact in the predicted evidence, Nov. 22, 2023. 2. It is supported by the first fact in the predicted evidence. 3. It is supported by the third fact in the predicted evidence. 4. No related facts can be found in the predicted evidence set.\n\nReturn the output in the exact format as specified in the examples, do not generate any additional output:\n\n"
    ]
)

ques_val_demo = "".join(open(os.path.join(root_dir, "templates/ques_evaluation_text.txt")).readlines())
justi_val_demo = "".join(open(os.path.join(root_dir, "templates/justi_evaluation_text.txt")).readlines())


def split_string_by_words(text, word_list):
    # Create a regex pattern with word boundaries for each word in the list
    pattern = r"(" + r"|".join(map(re.escape, word_list)) + r")"
    # Use re.split to split the text and keep the delimiters
    split_result = re.split(pattern, text)
    # Remove empty strings and strip spaces
    split_result = [s.strip() for s in split_result if s.strip()]
    return split_result


def gen_incontext_input_textonly(pred, ref, demos):
    texts = []
    texts.append(demos)
    texts.append("\n[PRED]: " + pred)
    texts.append("[REF]: " + ref)
    texts = "\n".join(texts)
    return texts


def score_extraction(feedback):
    pred_in_ref = feedback.split("[PRED in REF]: ")[-1].split("\n")[0].split(";")[0].strip()
    ref_in_pred = feedback.split("[REF in PRED]: ")[-1].split("\n")[0].split(";")[0].strip()
    if pred_in_ref.isdigit():
        pred_in_ref = int(pred_in_ref)
    else:
        pred_in_ref = 0
    if ref_in_pred.isdigit():
        ref_in_pred = int(ref_in_pred)
    else:
        ref_in_pred = 0
    score = {"ref_in_pred": ref_in_pred, "pred_in_ref": pred_in_ref}
    if len(feedback.split("[PRED in REF]: ")[-1].split("\n")[0].split(";")):
        score["detailed_ref_in_pred"] = ";".join(
            feedback.split("[REF in PRED]: ")[-1].split("\n")[0].split(";")[1:]
        ).strip()
        score["detailed_pred_in_ref"] = ";".join(
            feedback.split("[PRED in REF]: ")[-1].split("\n")[0].split(";")[1:]
        ).strip()
    return score


def seperate_text_val(
    gt_set, pred_set, path, eval_name, llm_name, mllm_name, save_num, debug_mode=False, eval_type=None
):
    # scores={user:{} for user in all_users_pred}
    demonstrations = open(os.path.join(path, "templates/ques_evaluation_text.txt")).readlines()
    demonstrations = "".join(demonstrations)

    if "gemini" in eval_name:
        from google import genai
        from private_info import API_keys
        from google.genai.types import HttpOptions

        model = genai.Client(http_options=HttpOptions(api_version="v1"), api_key=API_keys.GEMINI_API_KEY)
    elif "gemma" in eval_name:
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        llm = Gemma3ForConditionalGeneration.from_pretrained(
            eval_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(eval_name)
        model = {"model": llm.eval(), "processor": processor}

    raw_response = {}
    processed_response = {}
    for req_id in pred_set:
        if "gemini" in llm_name:
            pred = [row for k, row in enumerate(pred_set[req_id])]
        else:
            pred = [str(k + 1) + ". " + row for k, row in enumerate(pred_set[req_id])]
        gt = [str(k + 1) + ". " + row for k, row in enumerate(gt_set[req_id])]

        ref = " ".join(gt)
        pred = " ".join(pred)

        print("###", req_id, "###")
        print("GT evid:\n\t", ref)
        print("Pred evid:\n\t", pred)
        incontext_input = gen_incontext_input_textonly(pred, ref, demonstrations)
        if "gemini" in eval_name:
            response = model.models.generate_content(
                # model='gemini-2.5-pro-exp-03-25',
                model="gemini-2.0-flash-001",
                contents=incontext_input,
            )
            feedback = response.text
        elif "gemma" in eval_name:
            messages = [{"role": "user", "content": [{"type": "text", "text": incontext_input}]}]
            inputs = (
                model["processor"]
                .apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
                )
                .to(model["model"].device)
            )
            with torch.no_grad():
                generated_ids = model["model"].generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            feedback = model["processor"].batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

        processed_score = score_extraction(feedback)
        raw_response[req_id] = feedback
        processed_response[req_id] = processed_score
        if debug_mode:
            print(feedback, "\n\n")
    pkl.dump(
        raw_response,
        open(
            os.path.join(
                path,
                "open_evaluation",
                "intermediate_info/"
                + "_".join([llm_name, mllm_name])
                + "_val_text_"
                + str(save_num)
                + "_raw.pkl",
            ),
            "wb",
        ),
    )
    pkl.dump(
        processed_response,
        open(
            os.path.join(
                path,
                "open_evaluation",
                "intermediate_info/"
                + "_".join([llm_name, mllm_name])
                + "_val_text_"
                + str(save_num)
                + "_processed.pkl",
            ),
            "wb",
        ),
    )
    return


def gen_img_text_split(evid_context, pred=False):
    inputs = []
    for i, evid in enumerate(evid_context):
        evid_text = evid["text"] + " "
        if i == 0 and pred:
            evid_text = "[REF]: " + evid_text
        elif i == 0:
            evid_text = "[PRED]: " + evid_text
        evid_images = evid["images"]
        if len(evid_images) == 0:
            inputs.append((str(i + 1) + ". " + evid_text))
        else:
            img_token_list = re.findall(r"\[IMG_.*?\]", evid_text)  # [IMG_1], [IMG_2]...
            if len(img_token_list) == 0:
                inputs.append((str(i + 1) + ". " + evid_text))
            else:
                split_string = split_string_by_words(evid_text, img_token_list)
                for m, sp_str in enumerate(split_string):
                    if sp_str in img_token_list:
                        img_idx = re.findall(r"\d+", sp_str)[0]
                        inputs.append(Image.open(evid_images[int(img_idx) - 1]).convert("RGB"))
                    else:
                        if m == 0:
                            inputs.append(str(i + 1) + ". " + sp_str)
                        else:
                            inputs.append(sp_str)
    return inputs


def val_evid_idv(model, model_name, pred_evid, ref_evid, text_val, seperate_val):
    pred = [str(k + 1) + ". " + row["text"] for k, row in enumerate(pred_evid)]
    gt = [str(k + 1) + ". " + row["text"] for k, row in enumerate(ref_evid)]
    ref = ". ".join(gt)
    pred = ". ".join(pred)
    if text_val or seperate_val:
        # print ('GT evid:\n\t',ref)
        # print ('Pred evid:\n\t',pred)
        if seperate_val:
            # print ('Seperation!')
            incontext_input = gen_incontext_input_textonly(pred, ref, seperate_val_demo)
        else:
            incontext_input = gen_incontext_input_textonly(pred, ref, text_val_demo)
        if "gemini" in model_name:
            response = model.models.generate_content(model="gemini-2.0-flash-001", contents=incontext_input)
            feedback = response.text
        elif "gemma" in model_name:
            messages = [{"role": "user", "content": [{"type": "text", "text": incontext_input}]}]
            inputs = (
                model["processor"]
                .apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
                )
                .to(model["model"].device)
            )
            with torch.no_grad():
                generated_ids = model["model"].generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            feedback = model["processor"].batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    else:
        # consider inter-leaved image-text evaluation
        inputs = [joint_val_demo]
        ref_split = gen_img_text_split(ref_evid)
        pred_split = gen_img_text_split(pred_evid, pred=True)
        inputs.extend(ref_split)
        inputs.extend(pred_split)
        if "gemini" in model_name:
            response = model.models.generate_content(model="gemini-2.0-flash-001", contents=inputs)
            feedback = response.text
    processed_score = score_extraction(feedback)
    return feedback, processed_score


def compute_image_scores(model, model_name, pred_evid, ref_evid, score):
    prompt = "Given two sets of images, you need to score how similar they are, ranging from 0-10. The number of images could be different in image sets.\n"
    prompt += "[IMG_SET_1]:"
    ref_in_pred = re.findall(r"\(.*?\)", score["detailed_ref_in_pred"])
    pred_in_ref = re.findall(r"\(.*?\)", score["detailed_pred_in_ref"])
    print("ref in pred:", ref_in_pred, "\n pred in ref", pred_in_ref)
    image_scores = {"pred_in_ref": [], "ref_in_pred": []}
    # print (ref_in_pred)
    # print (pred_in_ref)
    for detail in pred_in_ref:
        info = detail[1:-1].split(",")
        # print ('pred in ref:',info)
        try:
            pred_idx = int(info[0].split("_")[-1])
            ref_idx = int(info[1].split("_")[-1])
            imgs_pred = pred_evid[pred_idx - 1]["images"]
            imgs_ref = ref_evid[ref_idx - 1]["images"]
            if len(imgs_pred) == 0 or len(imgs_ref) == 0:
                feedback = "10"
            else:
                if "gemini" in model_name:
                    inputs = [prompt]
                    for img in imgs_pred:
                        inputs.append(Image.open(img).convert("RGB"))
                    inputs.append("\n[IMG_SET_2]:")
                    for img in imgs_ref:
                        inputs.append(Image.open(img).convert("RGB"))
                    inputs.append("\nPlease generate your rating with one integer:")
                    response = model.models.generate_content(model="gemini-2.0-flash-001", contents=inputs)
                    feedback = response.text
                elif "gemma" in model_name:
                    messages = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}],
                        }
                    ]
                    for img in imgs_pred:
                        messages[0]["content"].append({"type": "image", "image": img})
                    messages[0]["content"].append({"type": "text", "text": "\n[IMG_SET_2]:"})
                    for img in imgs_ref:
                        messages[0]["content"].append({"type": "image", "image": img})
                    messages[0]["content"].append(
                        {"type": "text", "text": "\nPlease generate your rating with one integer:"}
                    )
                    inputs = (
                        model["processor"]
                        .apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt",
                        )
                        .to(model["model"].device)
                    )
                    with torch.no_grad():
                        generated_ids = model["model"].generate(**inputs, max_new_tokens=1024)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    feedback = model["processor"].batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True
                    )[0]
        except:
            print("##Edge case image!!")
            feedback = "10"
        image_scores["pred_in_ref"].append({"info": info, "score": feedback})
    for detail in ref_in_pred:
        info = detail[1:-1].split(",")
        # print ('ref in pred',info)
        try:
            pred_idx = int(info[1].split("_")[-1])
            ref_idx = int(info[0].split("_")[-1])
            imgs_pred = pred_evid[pred_idx - 1]["images"]
            imgs_ref = ref_evid[ref_idx - 1]["images"]
            if len(imgs_pred) == 0 or len(imgs_ref) == 0:
                feedback = "10"
            else:
                if "gemini" in model_name:
                    inputs = [prompt]
                    for img in imgs_pred:
                        inputs.append(Image.open(img).convert("RGB"))
                    inputs.append("\n[IMG_SET_2]:")
                    for img in imgs_ref:
                        inputs.append(Image.open(img).convert("RGB"))
                    inputs.append("\nPlease generate your rating with one integer:")
                    response = model.models.generate_content(model="gemini-2.0-flash-001", contents=inputs)
                    feedback = response.text
                elif "gemma" in model_name:
                    messages = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}],
                        }
                    ]
                    for img in imgs_pred:
                        messages[0]["content"].append({"type": "image", "image": img})
                    messages[0]["content"].append({"type": "text", "text": "\n[IMG_SET_2]:"})
                    for img in imgs_ref:
                        messages[0]["content"].append({"type": "image", "image": img})
                    messages[0]["content"].append(
                        {"type": "text", "text": "\nPlease generate your rating with one integer:"}
                    )
                    inputs = (
                        model["processor"]
                        .apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt",
                        )
                        .to(model["model"].device)
                    )
                    with torch.no_grad():
                        generated_ids = model["model"].generate(**inputs, max_new_tokens=1024)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    feedback = model["processor"].batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True
                    )[0]
        except:  # https://github.com/abril4416/AVerImaTeC?tab=readme-ov-file
            print("##Edge case image!!")
            feedback = "10"
        image_scores["ref_in_pred"].append({"info": info, "score": feedback})
    return image_scores


def textual_val_single(ref, pred, path, eval_name, model, eval_type="", debug_mode=False):
    if eval_type == "justification":
        val_demo = justi_val_demo
    elif eval_type == "question":
        val_demo = ques_val_demo
        if pred[0][0].isdigit() == False:
            pred = [str(k + 1) + ". " + row for k, row in enumerate(pred)]
        pred = " ".join(pred)
        ref = [str(k + 1) + ". " + row for k, row in enumerate(ref)]
        ref = " ".join(ref)

    incontext_input = gen_incontext_input_textonly(pred, ref, val_demo)
    if "gemini" in eval_name:
        response = model.models.generate_content(
            # model='gemini-2.5-pro-exp-03-25',
            model="gemini-2.0-flash-001",
            contents=incontext_input,
        )
        feedback = response.text
    elif "gemma" in eval_name:
        messages = [{"role": "user", "content": [{"type": "text", "text": incontext_input}]}]
        inputs = (
            model["processor"]
            .apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            )
            .to(model["model"].device)
        )
        with torch.no_grad():
            generated_ids = model["model"].generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        feedback = model["processor"].batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    processed_score = score_extraction(feedback)

    if debug_mode:
        print(eval_type)
        print(processed_score)
        print(feedback, "\n\n")

    return feedback, processed_score
