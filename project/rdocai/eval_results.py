import os
import json
from itertools import product

from tqdm.auto import tqdm
from nltk import edit_distance

model = "qwenvl"

results_dir = f"/data1/home/ohs/workspace/rai/generated_jsons/{model}"
results_jsons = [f for f in os.listdir(results_dir) if f.endswith('.json')]
idxes = sorted(list(set([f.split('_')[0] for f in results_jsons])), key=lambda x: int(x))


cuts = ["half_cut", "third_cut", "quarter_cut"]
tan_idxes = [0, 1, 2, 3]

results_dict = {}
for idx in tqdm(idxes):
    original_json = os.path.join(results_dir, f"{idx}_original.json")
    original_json = open(original_json, 'r').read().strip()

    sub_res = {}
    for cut, tan_idx in product(cuts, tan_idxes):
        generated_json = os.path.join(results_dir, f"{idx}_{cut}_{tan_idx}.json")
        generated_json = open(generated_json, 'r').read().strip()

        ed = edit_distance(original_json, generated_json)
        normed_ed = ed / len(original_json)

        sub_res["_".join([cut, str(tan_idx)])] = {
            "generated_json": generated_json,
            "normed_ed": normed_ed
            }
        
    results_dict[idx] = sub_res

with open(os.path.join("/data1/home/ohs/workspace/rai/eval_results", f"{model}_results.json"), 'w') as f:
    json.dump(results_dict, f, indent=4, ensure_ascii=False)