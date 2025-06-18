import os
import re
import json

from tqdm.auto import tqdm

from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

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

model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device="cuda")
preprocessor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

imgs_dir = "/data1/home/ohs/workspace/rai/rotated_images"
save_base_dir = "/data1/home/ohs/workspace/rai/generated_jsons/hcx"
os.makedirs(save_base_dir, exist_ok=True)

imgs = [f for f in os.listdir(imgs_dir) if f.endswith(".png")]
imgs.sort(key=lambda x: int(x.split("_")[0]))
for idx_img, img_fname in enumerate(tqdm(imgs)):
    img_path = os.path.join(imgs_dir, img_fname)
    vlm_chat = [
        {
            "role": "system",
            "content":{
                "type": "text",
                "text": "You are helpful assistant!"
            }
        }, 
        {
            "role": "user",
            "content":{
                "type": "image",
                "filename": f"recipt_{idx_img:03d}.png",
                "image": img_path,
            }
        },
        {
            "role": "user",
            "content":{
                "type": "text",
                "text": user_prompt
            }
        }, 
    ]

    new_vlm_chat, all_images, is_video_list = preprocessor.load_images_videos(vlm_chat)
    preprocessed = preprocessor(all_images, is_video_list=is_video_list)
    input_ids = tokenizer.apply_chat_template(
            new_vlm_chat, return_tensors="pt", tokenize=True, add_generation_prompt=True,
    )
    
    output_ids = model.generate(
            input_ids=input_ids.to(device="cuda"),
            max_length=None,
            max_new_tokens=8192,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.0,
            **preprocessed,
    )

    generated_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    parsed_output = parse_output(generated_output)

    with open(os.path.join(save_base_dir, os.path.basename(img_path).replace(".png", ".json")), "w", encoding="utf-8") as f:
        f.write(parsed_output)