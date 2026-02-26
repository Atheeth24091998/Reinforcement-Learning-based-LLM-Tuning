from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name: str, device_map="auto"):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
    return model, tok