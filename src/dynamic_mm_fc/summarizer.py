import torch
import numpy as np
import nltk


class Summarize_Model:
    def __init__(self, llm, llm_name, debug):
        self.llm_model = llm
        self.llm_name = llm_name

        self.debug = debug
        self.prompt_head = "You are a helpful assistant to summarize a textual justification in one or two sentences. Here is the justification: %s. Please generate your summarization:"

    def summarize(self, justification):
        justi = " ".join(justification.split("\n"))
        inputs = self.prompt_head % justi
        if "gemini" in self.llm_name:
            response = self.llm_model.models.generate_content(model=self.llm_name, contents=[inputs])
            response = response.text
            plan = response.strip()
        elif self.llm_name == "qwen":
            msg = inputs
            messages = [
                {"role": "system", "content": "You are a helpful assistant!"},
                {"role": "user", "content": msg},
            ]
            text = self.llm_model["tokenizer"].apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self.llm_model["tokenizer"]([text], return_tensors="pt").to(
                self.llm_model["model"].device
            )
            generated_ids = self.llm_model["model"].generate(**model_inputs, max_new_tokens=256)
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.llm_model["tokenizer"].batch_decode(generated_ids, skip_special_tokens=True)[0]
            plan = response.split(":")[-1]
        elif self.llm_name == "gemma":
            msg = inputs
            messages = [{"role": "user", "content": [{"type": "text", "text": msg}]}]
            inputs = (
                self.llm_model["processor"]
                .apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    # , padding="max_length", max_length=4096, truncation=True
                )
                .to(self.llm_model["model"].device)
            )
            with torch.no_grad():
                generated_ids = self.llm_model["model"].generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.llm_model["processor"].batch_decode(
                generated_ids_trimmed, skip_special_tokens=True
            )[0]
            plan = response.split(":")[-1]
        return plan
