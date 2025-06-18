import os
import re
import json

from tqdm.auto import tqdm

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def parse_output(text):
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
        json_str = re.sub(r'\s+', ' ', json_str)
        try:
            json_data = json.loads(json_str)
            return json.dumps(json_data, indent=4, ensure_ascii=False)
        except json.JSONDecodeError:
            return json_str
    else:
        return text

user_prompt = """
Please return the contents of this receipt image as structured JSON data.
""".strip()

model_name = "/data1/home/ohs/workspace/rai/rdocai/Qwen/Qwen2.5-VL-3B-Instruct-sft-llava-instruct-mix-vsft/checkpoint-164"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="cuda"
)
processor = AutoProcessor.from_pretrained(model_name)

imgs_dir = "/data1/home/ohs/workspace/rai/rotated_images"
save_base_dir = "/data1/home/ohs/workspace/rai/trained/rotated_2/generated_jsons/qwenvl"
os.makedirs(save_base_dir, exist_ok=True)

imgs = [f for f in os.listdir(imgs_dir) if f.endswith(".png")]
imgs.sort(key=lambda x: int(x.split("_")[0]))
for idx_img, img_fname in enumerate(tqdm(imgs)):
    img_path = os.path.join(imgs_dir, img_fname)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                },
                {"type": "text", "text": user_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    generated_ids = model.generate(**inputs, max_new_tokens=8192, do_sample=False,temperature=None)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    parsed_output = parse_output(output_text)

    with open(os.path.join(save_base_dir, os.path.basename(img_path).replace(".png", ".json")), "w", encoding="utf-8") as f:
        f.write(parsed_output)