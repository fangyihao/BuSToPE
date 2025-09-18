import json
import os
import subprocess
ropes = ['m_rope', 'videorope', 'tad_rope', 'vanilla_rope']

for rope in ropes:
    args = dict(
      deepspeed="ds_z3_config.json",
      model_name_or_path="models--Qwen--Qwen2-VL-2B-Instruct",            # use Qwen/Qwen2-VL model
      stage="sft",                                               # do supervised fine-tuning
      total_pixels=6272000,
      video_maxlen=128,
      do_train=True,
      finetuning_type="full",
      dataset="llava_videos_330k_split0,llava_videos_330k_split1,llava_videos_330k_split2",               # use llava videos datasets
      seed=42,
      template="qwen2_vl",                                         # use qwen2_vl prompt template
      cutoff_len=8200,
      overwrite_cache=True,
      preprocessing_num_workers=32,
      output_dir=f"checkpoints/models--Qwen--Qwen2-VL-2B-Instruct/{rope}",                                  # the path to save model checkpoints
      num_train_epochs=1.0,                                   # the number of training epochs
      logging_steps=1,                                      # log every step
      save_steps=1,                                       # save checkpoint every 5 steps
      save_total_limit=1,
      plot_loss=True,
      overwrite_output_dir=True,
      per_device_train_batch_size=1,                      # the micro batch size
      gradient_accumulation_steps=128,                     # the gradient accumulation steps
      learning_rate=1.0e-5,                               # the learning rate
      lr_scheduler_type="cosine",                                # use cosine learning rate scheduler
      warmup_ratio=0.1,                                          # use warmup scheduler
      bf16=True,                                                 # use bfloat16 mixed precision training
      ddp_timeout=180000000,
      val_size=1,
      per_device_eval_batch_size=1,
      eval_strategy="steps",
      eval_steps=1,
      flash_attn="fa2",
      #gradient_checkpointing=True,
      resume_from_checkpoint=f"checkpoints/models--Qwen--Qwen2-VL-2B-Instruct/{rope}/checkpoint-180",
      which_rope= rope,  
      report_to="none",                                          # disable wandb logging
      tokenized_path=f"tokenized/models--Qwen--Qwen2-VL-2B-Instruct/{rope}",
      #buffer_size=512,
      #preprocessing_batch_size=512,
      #streaming=True,
      #max_steps=1,
      #accelerator_config=dict(
      #  dispatch_batches=False
      #)
    )

    json.dump(args, open(f"./train_qwen2-vl_{rope}.json", "w", encoding="utf-8"), indent=2)
    os.system(f'mkdir -p tokenized/models--Qwen--Qwen2-VL-2B-Instruct/{rope}')
    os.system(f'mkdir -p checkpoints/models--Qwen--Qwen2-VL-2B-Instruct/{rope}')
'''
import json
import os
for i in range(3):
    #os.system(f'curl -o ./data/llava-video-other_times-300k-2-3mins-30k-split110k_{i}_remove_broken.json -L https://huggingface.co/datasets/Wiselnn/VideoRoPE/resolve/main/llava-video-other_times-300k-2-3mins-30k-split110k_{i}_remove_broken.json?download=true')
    os.system(f'cp ../llava-video-other_times-300k-2-3mins-30k-split110k_{i}_remove_broken.json ./data/llava-video-other_times-300k-2-3mins-30k-split110k_{i}_remove_broken.json')
    with open(f'./data/llava-video-other_times-300k-2-3mins-30k-split110k_{i}_remove_broken.json', 'r') as f:
        data = json.load(f)
    #data = data[-10240:]
    for row in data:
        row['videos'][0] = row['videos'][0].replace('qian:s3://IXC3_data/LLaVA_Video/', '/data1/yihao/llava-video/')
        # print(row['videos'][0])
    with open(f'./data/llava-video-other_times-300k-2-3mins-30k-split110k_{i}_remove_broken.json', 'w') as f:
        json.dump(data, f, indent=4)



with open(f'./data/dataset_info.json', 'r') as f:
    data = json.load(f)

llava_video_other = {"llava_videos_330k_split0": {
    "file_name": "llava-video-other_times-300k-2-3mins-30k-split110k_0_remove_broken.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "videos": "videos"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  },
  "llava_videos_330k_split1": {
    "file_name": "llava-video-other_times-300k-2-3mins-30k-split110k_1_remove_broken.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "videos": "videos"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  },
  "llava_videos_330k_split2": {
    "file_name": "llava-video-other_times-300k-2-3mins-30k-split110k_2_remove_broken.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "videos": "videos"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }}

data.update(llava_video_other)

with open(f'./data/dataset_info.json', 'w') as f:
    json.dump(data, f, indent=4)
'''
rope = ropes[1]
subprocess.run(f'source /data1/yihao/videorope_venv/bin/activate && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup llamafactory-cli train train_qwen2-vl_{rope}.json > {rope}_output.log 2>&1 &', shell=True, executable="/bin/bash")
