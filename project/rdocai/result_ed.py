import json
from pathlib import Path
from itertools import product

from tqdm.auto import tqdm
from nltk import edit_distance

from datasets import load_dataset

def make_generation_res(generation_dir, gts, save_fname):

    cut = ["half_cut", "third_cut", "quarter_cut"]
    extent = ["0", "1", "2", "3"]

    with open(save_fname, "w", encoding="utf-8") as f:
        generation_res = []

        for i in tqdm(range(len(gts))):
            gt = gts[i]

            result_for_sample = {"grount_truth": gt}

            original_key = f"{i:03d}_original"
            original_path = generation_dir / f"{original_key}.json"

            try:
                original_text = original_path.read_text()
                try:
                    json.loads(original_text)
                    result_for_sample[original_key] = {
                        "is_json_structured": True,
                        "generated_text": json.dumps(original_text, ensure_ascii=False),
                        "edit_distance_to_gt": edit_distance(original_text, gt)
                    }
                except:
                    print(f"Error reading JSON from {original_path}")
                    result_for_sample[original_key] = {
                        "is_json_structured": False,
                        "generated_text": original_text,
                        "edit_distance_to_gt": edit_distance(original_text, gt)
                    }
            except:
                raise FileNotFoundError(f"Original generation file not found: {original_path}")


            result_for_sample[original_key]
            for c, e in product(cut, extent):

                generation_key = f"{i:03d}_{c}_{e}"
                result_for_sample[generation_key] = {}
                generation_path = generation_dir / f"{generation_key}.json"
                try:
                    generated_text = generation_path.read_text()
                    try:
                        json.loads(generated_text)
                        result_for_sample[generation_key]["is_json_structured"] = True
                        result_for_sample[generation_key]["generated_text"] = json.dumps(generated_text, ensure_ascii=False)
                        result_for_sample[generation_key]["edit_distance_to_gt"] = edit_distance(generated_text, gt)
                        result_for_sample[generation_key]["edit_distance_to_original"] = edit_distance(generated_text, result_for_sample[original_key]["generated_text"])
                    except:
                        print(f"Error reading JSON from {generation_path}")
                        result_for_sample[generation_key]["is_json_structured"] = False
                        result_for_sample[generation_key]["generated_text"] = generated_text
                        result_for_sample[generation_key]["edit_distance_to_gt"] = edit_distance(generated_text, gt)
                        result_for_sample[generation_key]["edit_distance_to_original"] = edit_distance(generated_text, result_for_sample[original_key]["generated_text"])
                except:
                    raise FileNotFoundError(f"Generation file not found: {generation_path}")
            generation_res.append(result_for_sample)
            f.write(json.dumps(result_for_sample, ensure_ascii=False) + "\n")
        
    return generation_res

ds = load_dataset("naver-clova-ix/cord-v2", split="test")

gts = [sample["ground_truth"] for sample in ds]
gts = [json.loads(gt)["gt_parse"] for gt in gts]
gts = [json.dumps(gt, ensure_ascii=False) for gt in gts]

save_base_dir = Path("/data1/home/ohs/workspace/rai/whole_result")
save_base_dir.mkdir(parents=True, exist_ok=True)

gen_save_pairs = []

hcx_generation_dir = Path("/data1/home/ohs/workspace/rai/original/generated_jsons/hcx")
hcx_save_fname = save_base_dir / "hcx_generation_res.jsonl"
gen_save_pairs.append((hcx_generation_dir, hcx_save_fname))


qwenvl_generation_dir = Path("/data1/home/ohs/workspace/rai/original/generated_jsons/qwenvl")
qwenvl_save_fname = save_base_dir / "qwenvl_generation_res.jsonl"
gen_save_pairs.append((qwenvl_generation_dir, qwenvl_save_fname))

qwenvl_train_only_original_generation_dir = Path("/data1/home/ohs/workspace/rai/trained/only/generated_jsons/qwenvl")
qwenvl_train_only_save_fname = save_base_dir / "qwenvl_train_only_generation_res.jsonl"
gen_save_pairs.append((qwenvl_train_only_original_generation_dir, qwenvl_train_only_save_fname))

qwenvl_train_1epoch_generation_dir = Path("/data1/home/ohs/workspace/rai/trained/rotated_1/generated_jsons/qwenvl")
qwenvl_train_1epoch_save_fname = save_base_dir / "qwenvl_train_1epoch_generation_res.jsonl"
gen_save_pairs.append((qwenvl_train_1epoch_generation_dir, qwenvl_train_1epoch_save_fname))

qwenvl_train_2epoch_generation_dir = Path("/data1/home/ohs/workspace/rai/trained/rotated_2/generated_jsons/qwenvl")
qwenvl_train_2epoch_save_fname = save_base_dir / "qwenvl_train_2epoch_generation_res.jsonl"
gen_save_pairs.append((qwenvl_train_2epoch_generation_dir, qwenvl_train_2epoch_save_fname))

qwenvl_train_3epoch_generation_dir = Path("/data1/home/ohs/workspace/rai/trained/rotated_3/generated_jsons/qwenvl")
qwenvl_train_3epoch_save_fname = save_base_dir / "qwenvl_train_3epoch_generation_res.jsonl"
gen_save_pairs.append((qwenvl_train_3epoch_generation_dir, qwenvl_train_3epoch_save_fname))