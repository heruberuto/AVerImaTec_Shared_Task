import os

root_dir = os.path.abspath("..")
import torch


def gen_incontext_input(ques, ans, demos):
    texts = []
    texts.append(demos)
    texts.append("[QUES]: " + ques)
    texts.append("[ANS]: " + ans)
    texts.append("[STAT]:")
    texts = "\n".join(texts)
    return texts


def qa_to_evid(ques, ans, llm, llm_name):
    # loading demonstrations
    demonstrations = open(os.path.join(root_dir, "templates/qa_to_evid_demos.txt")).readlines()
    demonstrations = "\n".join(demonstrations)
    incontext_input = gen_incontext_input(ques, ans, demonstrations)
    if "gemini" in llm_name:
        response = llm.models.generate_content(model=llm_name, contents=incontext_input)
        statement = response.text
        statement = statement.replace("[STAT]:", "").strip()
    elif "gemma" in llm_name:
        messages = [{"role": "user", "content": [{"type": "text", "text": incontext_input}]}]
        inputs = (
            llm["processor"]
            .apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            )
            .to(llm["model"].device)
        )
        with torch.no_grad():
            generated_ids = llm["model"].generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = llm["processor"].batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        statement = response.replace("[STAT]:", "").strip()
    return statement
