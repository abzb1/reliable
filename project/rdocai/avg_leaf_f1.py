import json
from collections import deque

def leaf_vals(obj):
    leaf_vals = []

    if isinstance(obj, dict):
        to_go = deque(list(obj.values()))
    elif isinstance(obj, list):
        to_go = deque(obj)

    while to_go:
        v = to_go.popleft()
        if isinstance(v, dict):
            to_go.extend(v.values())
        elif isinstance(v, list):
            to_go.extend(v)
        else:
            leaf_vals.append(v)

    return leaf_vals

def f1_leaf(target, generated):
    t_vals = set(target)
    g_vals = set(generated)

    tp = len(t_vals & g_vals)
    precision = tp / len(g_vals) if g_vals else 1.0
    recall    = tp / len(t_vals) if t_vals else 1.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1

def get_json_leaf_node_f1(gen_res):
    ks = ['original', 'half_cut_0', 'half_cut_1', 'half_cut_2', 'half_cut_3', 'third_cut_0', 'third_cut_1', 'third_cut_2', 'third_cut_3', 'quarter_cut_0', 'quarter_cut_1', 'quarter_cut_2', 'quarter_cut_3']

    leaf_f1_to_gt = {k: 0 for k in ks}
    leaf_f1_to_original = {k: 0 for k in ks}
    for key in ks:
        gt_count = 0
        original_count = 0
        for q_idx in range(len(gen_res)):
            gt = gen_res[q_idx]["ground_truth"]
            result = gen_res[q_idx][f"{q_idx:03d}_{key}"]
            
            if result["is_json_structured"]:
                result_text = result["generated_text"]
            else:
                continue

            gt_leafs = leaf_vals(json.loads(gt))
            result_leafs = leaf_vals(json.loads(result_text))

            if key != "original":
                original_result = gen_res[q_idx][f"{q_idx:03d}_original"]
                if original_result["is_json_structured"]:
                    original_text = original_result["generated_text"]
                    original_leafs = leaf_vals(json.loads(original_text))
                    leaf_f1_to_original[key] += f1_leaf(original_leafs, leaf_vals(json.loads(result_text)))
                    original_count += 1

            f1 = f1_leaf(gt_leafs, result_leafs)
            leaf_f1_to_gt[key] += f1
            gt_count += 1
        if gt_count > 0:
            leaf_f1_to_gt[key] /= gt_count
        if original_count > 0:
            leaf_f1_to_original[key] /= original_count

    return leaf_f1_to_gt, leaf_f1_to_original

hcx_res = "/data1/home/ohs/workspace/rai/whole_result/hcx_generation_res.jsonl"
with open(hcx_res, 'r') as f:
    hcx_res = [json.loads(line) for line in f.readlines()]

qwenvl_res = "/data1/home/ohs/workspace/rai/whole_result/qwenvl_generation_res.jsonl"
with open(qwenvl_res, 'r') as f:
    qwenvl_res = [json.loads(line) for line in f.readlines()]

qwenvl_trained_res = "/data1/home/ohs/workspace/rai/whole_result/qwenvl_train_only_save_fname.jsonl"
with open(qwenvl_trained_res, 'r') as f:
    qwenvl_trained_res = [json.loads(line) for line in f.readlines()]

gen_res1 = "/data1/home/ohs/workspace/rai/whole_result/qwenvl_train_1epoch_generation_res.jsonl"
with open(gen_res1, 'r') as f:
    gen_res1 = [json.loads(line) for line in f.readlines()]

gen_res2 = "/data1/home/ohs/workspace/rai/whole_result/qwenvl_train_2epoch_generation_res.jsonl"
with open(gen_res2, 'r') as f:
    gen_res2 = [json.loads(line) for line in f.readlines()]

gen_res3 = "/data1/home/ohs/workspace/rai/whole_result/qwenvl_train_3epoch_generation_res.jsonl"
with open(gen_res3, 'r') as f:
    gen_res3 = [json.loads(line) for line in f.readlines()]

print("hcx_res")
leaf_f1_gt, leaf_f1_origin = get_json_leaf_node_f1(hcx_res)
print("leaf_f1_gt", leaf_f1_gt.values())
print("leaf_f1_origin", leaf_f1_origin.values())

print("qwenvl_res")
leaf_f1_gt, leaf_f1_origin = get_json_leaf_node_f1(qwenvl_res)
print("leaf_f1_gt", leaf_f1_gt.values())
print("leaf_f1_origin", leaf_f1_origin.values())

print("qwenvl_trained_res")
leaf_f1_gt, leaf_f1_origin = get_json_leaf_node_f1(qwenvl_trained_res)
print("leaf_f1_gt", leaf_f1_gt.values())
print("leaf_f1_origin", leaf_f1_origin.values())

print("gen_res1")
leaf_f1_gt, leaf_f1_origin = get_json_leaf_node_f1(gen_res1)
print("leaf_f1_gt", leaf_f1_gt.values())
print("leaf_f1_origin", leaf_f1_origin.values())

print("gen_res2")
leaf_f1_gt, leaf_f1_origin = get_json_leaf_node_f1(gen_res2)
print("leaf_f1_gt", leaf_f1_gt.values())
print("leaf_f1_origin", leaf_f1_origin.values())

print("gen_res3")
leaf_f1_gt, leaf_f1_origin = get_json_leaf_node_f1(gen_res3)
print("leaf_f1_gt", leaf_f1_gt.values())
print("leaf_f1_origin", leaf_f1_origin.values())