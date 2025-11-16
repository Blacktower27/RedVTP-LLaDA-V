from transformers.generation import stopping_criteria
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from llava.cache import dLLMCache, dLLMCacheConfig
from llava.hooks import register_cache_LLaDA_V
from dataclasses import asdict
from llava.hooks.fast_dllm_hook import register_fast_dllm_hook, unregister_fast_dllm_hook

from PIL import Image
import requests
import copy
import torch
import time

import sys
import warnings

# prompt_interval_steps = 25
# gen_interval_steps = 7
prompt_interval_steps = 25
gen_interval_steps = 7
transfer_ratio = 0.25
use_fast_dllm = False  # using fast-dLLM (https://github.com/NVlabs/Fast-dLLM) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 6s to generate 128 tokens.
use_dllm_cache = False  # using dLLM-Cache(https://github.com/maomaocun/dLLM-cache) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 25s to generate 128 tokens.

warnings.filterwarnings("ignore")
pretrained = "GSAI-ML/LLaDA-V"

model_name = "llava_llada"
device = "cuda:5"
device_map = "cuda:5"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, attn_implementation="sdpa", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.to(device)
model.eval()
# image = Image.open("temp_cc.jpg")
image = Image.open("sample_image/realworld7.jpg").convert("RGB")
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "llava_llada" 
# question = DEFAULT_IMAGE_TOKEN + "\nPlease describe the image in detail."
# question = DEFAULT_IMAGE_TOKEN + "\nAre the lights red? Options: A. Yes B. No Please answer directly with only the letter of the correct option and nothing else."
question = DEFAULT_IMAGE_TOKEN + "\nIs the left turning arrow green?"

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

model.eval()
if use_fast_dllm:
    register_fast_dllm_hook(model)
    print("Testing with Fast dLLM hook enabled")
elif use_dllm_cache:
    dLLMCache.new_instance(
        **asdict(
            dLLMCacheConfig(
                prompt_interval_steps=prompt_interval_steps,
                gen_interval_steps=gen_interval_steps,
                transfer_ratio=transfer_ratio,
            )
        )
    )
    register_cache_LLaDA_V(model, "model.layers")
    print("Testing with cache enabled")
else:
    print("Testing without cache")

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]





start_time = time.time()
# cont = model.generate(
#     input_ids,
#     images=image_tensor,
#     image_sizes=image_sizes,
#     steps=128, gen_length=128, block_length=128, tokenizer=tokenizer, stopping_criteria=['<|eot_id|>'], 
#     prefix_refresh_interval=32,
#     threshold=1,
# )
#infovqa setting
cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    steps=16, gen_length=32, block_length=32, tokenizer=tokenizer, stopping_criteria=['<|eot_id|>'], 
    prefix_refresh_interval=32,
    threshold=1,
)
# cont = model.generate(
#     input_ids,
#     images=image_tensor,
#     image_sizes=image_sizes,
#     steps=2, gen_length=2, block_length=2, tokenizer=tokenizer, stopping_criteria=['<|eot_id|>'], 
#     prefix_refresh_interval=32,
#     threshold=1,
# )
end_time = time.time()
generation_time = end_time - start_time
print(f"Generation time: {generation_time:.4f} seconds")

# print(cont)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=False)
print(text_outputs)

# import torch
# from PIL import Image, ImageDraw
# import matplotlib.pyplot as plt
# import numpy as np

# # ==== 路径 ====
# save_path = "/data3/cli78217/Jingqi_workspace/diffusion_VLM/LLaDA-V/train/visual_prune_indices.pt"
# # image_path = "test.jpg"

# # ==== 读取索引 ====
# data = torch.load(save_path)
# keep_idx = data["keep_global_idx"]
# vis_start, vis_end = data["vis_start"], data["vis_end"]

# # ==== 修正索引 ====
# keep_idx = keep_idx[(keep_idx >= vis_start) & (keep_idx < vis_end)]

# # ==== 计算网格 ====
# total_vis = vis_end - vis_start
# grid = int(np.sqrt(total_vis))

# # ==== 生成掩码 ====
# mask = torch.zeros(total_vis, dtype=torch.bool)
# mask[(keep_idx - vis_start).clamp(0, total_vis - 1)] = True

# # ==== 读图并拉伸成正方形 ====
# img = image
# L = max(img.size)
# img_square = img.resize((L, L), Image.Resampling.LANCZOS)

# # ==== 绘制 ====
# draw = ImageDraw.Draw(img_square)
# cell = L / grid
# for i in range(total_vis):
#     x, y = i % grid, i // grid
#     if not mask[i]:
#         # 不透明深灰色遮罩
#         draw.rectangle([x*cell, y*cell, (x+1)*cell, (y+1)*cell],
#                        outline=(50, 50, 50),
#                        fill=(80, 80, 80))

# # ==== 显示与保存 ====
# plt.imshow(img_square)
# plt.axis("off")
# plt.title("Gray = pruned")
# plt.show()
# img_square.save("visual_prune_overlay.png")
# print("Saved to visual_prune_overlay.png")

# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image, ImageDraw
# import os

# # ==== 参数 ====
# base_path = "/data3/cli78217/Jingqi_workspace/diffusion_VLM/LLaDA-V/train"
# num_steps = 16-1

# # ==== 加载原图并变为正方形 ====
# img = image
# L = max(img.size)
# img_square = img.resize((L, L), Image.Resampling.LANCZOS)
# total_vis = None

# for step in range(num_steps):
#     data = torch.load(os.path.join(base_path, f"vis_step_{step:02d}.pt"))
#     vis_importance = data["vis_importance"]
#     vis_start, vis_end = data["vis_start"], data["vis_end"]
#     total_vis = vis_end - vis_start

#     # ==== 归一化 + 对比增强 ====
#     imp = vis_importance.detach().cpu()
#     imp = (imp - imp.min()) / (imp.max() + 1e-6)
#     imp = imp ** 0.5   # 提升亮部权重（可调 0.3~0.7）

#     grid = int(np.sqrt(total_vis))
#     imp_2d = imp[:grid * grid].reshape(grid, grid).numpy()

#     # ==== 转换为热力图并增强饱和度 ====
#     cmap = plt.cm.jet(imp_2d)[:, :, :3]  # 取RGB部分
#     cmap = (cmap * 255).astype(np.uint8)
#     imp_img = Image.fromarray(cmap)
#     imp_img = imp_img.resize((L, L), Image.Resampling.LANCZOS)

#     # ==== 增强可视化叠加 ====
#     base = img_square.convert("RGBA")
#     heat = imp_img.convert("RGBA")

#     # 调整叠加比例（0.6 显著，0.4 柔和）
#     blended = Image.blend(base, heat, alpha=0.6)

#     # ==== 绘制并保存 ====
#     plt.imshow(blended)
#     plt.axis("off")
#     plt.title(f"Visual Importance Step {step}", fontsize=10, weight="bold")
#     plt.savefig(os.path.join(base_path, f"vis_importance_step_{step:02d}.png"), bbox_inches="tight", dpi=300)
#     plt.close()



import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import os

# ==== 参数 ====
base_path = "/data3/cli78217/Jingqi_workspace/diffusion_VLM/LLaDA-V/train"
num_steps = 16-1
keep_ratio = 0.5  # 保留最重要 token 的比例

# ==== 加载原图 ====
img = image
L = max(img.size)
img_square = img.resize((L, L), Image.Resampling.LANCZOS)

for step in range(num_steps):
    data = torch.load(os.path.join(base_path, f"vis_step_{step:02d}.pt"))
    vis_importance = data["vis_importance"].detach().cpu()
    vis_start, vis_end = data["vis_start"], data["vis_end"]
    total_vis = vis_end - vis_start
    grid = int(np.sqrt(total_vis))

    # ==== 归一化 ====
    vis_importance = (vis_importance - vis_importance.min()) / (vis_importance.max() + 1e-6)

    # ==== Top-K 保留索引 ====
    keep_n = max(1, int(total_vis * keep_ratio))
    _, keep_local_idx = torch.topk(vis_importance, k=keep_n, largest=True)
    keep_local_idx = torch.sort(keep_local_idx).values
    keep_idx = keep_local_idx + vis_start  # 转为全局索引

    # ==== 生成掩码 ====
    mask = torch.zeros(total_vis, dtype=torch.bool)
    mask[(keep_idx - vis_start).clamp(0, total_vis - 1)] = True

    # ==== 复制原图并绘制灰色遮罩 ====
    img_masked = img_square.copy()
    draw = ImageDraw.Draw(img_masked)
    cell = L / grid

    for i in range(total_vis):
        if not mask[i]:
            x, y = i % grid, i // grid
            draw.rectangle(
                [x * cell, y * cell, (x + 1) * cell, (y + 1) * cell],
                outline=(50, 50, 50),
                fill=(80, 80, 80)
            )

    # ==== 显示与保存 ====
    plt.imshow(img_masked)
    plt.axis("off")
    # plt.title(f"Step {step}: kept {int(keep_ratio * 100)}%", fontsize=10)
    plt.savefig(os.path.join(base_path, f"vis_pruned_step_{step:02d}.png"), bbox_inches="tight", dpi=300)
    plt.close()

print(f"Saved {num_steps} pruned visualizations (keep_ratio={keep_ratio}).")


