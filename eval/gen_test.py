import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['VLLM_ATTENTION_BACKEND']='FLASH_ATTN'

from vllm import LLM, SamplingParams
#from modeling_videorope import Qwen2VLForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration
from transformers import AutoProcessor
import torch

model_path = "/data/videorope_2/LLaMA-Factory/checkpoints/models--Qwen--Qwen2-VL-2B-Instruct/videorope"
context_length = 8192
def run(vllm_backend=False):
    if vllm_backend:
        llm = LLM(model_path,
                max_model_len=context_length+1536,
                limit_mm_per_prompt={"video": 10},
                gpu_memory_utilization=0.8,
                #pipeline_parallel_size=4
                )
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, 
                                                            device_map="auto",
                                                            torch_dtype=torch.bfloat16, 
                                                            attn_implementation="flash_attention_2"
                                                            )
        processor = AutoProcessor.from_pretrained(model_path)
        model = model.eval()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello, who is he president of the United States?"}
                ]
            },
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, return_tensors="pt")
        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=500)
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        print(generated_texts[0])

if __name__ == '__main__':
    run()