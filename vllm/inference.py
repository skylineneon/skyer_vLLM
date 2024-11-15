import torch
from vllm import LLM, SamplingParams

prompts = [
    "你好",
    "你是谁",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(
    temperature=0.7, 
    top_p=0.95,
    stop_token_ids=[3,])


_model_id = "/root/workspace/skyer_huggingface/cache/skyer_back"

llm = LLM(model=_model_id, 
          dtype=torch.float16, 
          trust_remote_code=True)


outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"{prompt!r} {generated_text!r}")
