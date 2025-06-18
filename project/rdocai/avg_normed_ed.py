import json

def get_avg_edit_distance(gen_res):
    ks = ['original', 'half_cut_0', 'half_cut_1', 'half_cut_2', 'half_cut_3', 'third_cut_0', 'third_cut_1', 'third_cut_2', 'third_cut_3', 'quarter_cut_0', 'quarter_cut_1', 'quarter_cut_2', 'quarter_cut_3']

    avg_normed_ed_to_gt = {key: 0 for key in ks}
    avg_normed_ed_to_original = {key: 0 for key in ks}
    for key in ks:

        for q_idx in range(len(gen_res)):
            gt = gen_res[q_idx]["ground_truth"]
            len_gt = len(gt)
            result = gen_res[q_idx][f"{q_idx:03d}_{key}"]
            
            if result["is_json_structured"]:
                ed_to_gt = result["edit_distance_to_gt"]
                normed_ed_to_gt = ed_to_gt / len_gt
                normed_ed_to_gt = min(1.0, normed_ed_to_gt)
                normed_ed_to_gt = normed_ed_to_gt / len(gen_res)

                avg_normed_ed_to_gt[key] += normed_ed_to_gt

                if key != "original":
                    ed_to_original = result["edit_distance_to_original"]
                    normed_ed_to_original = ed_to_original / len(gen_res[q_idx][f"{q_idx:03d}_original"]["generated_text"])
                    normed_ed_to_original = min(1.0, normed_ed_to_original)
                    normed_ed_to_original = normed_ed_to_original / len(gen_res)
                    avg_normed_ed_to_original[key] += normed_ed_to_original

    return avg_normed_ed_to_gt, avg_normed_ed_to_original

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

ed_to_gt, ed_to_origin = get_avg_edit_distance(hcx_res)
print(ed_to_gt.values())
print(ed_to_origin.values())

ed_to_gt, ed_to_origin = get_avg_edit_distance(qwenvl_res)
print(ed_to_gt.values())
print(ed_to_origin.values())

ed_to_gt, ed_to_origin = get_avg_edit_distance(qwenvl_trained_res)
print(ed_to_gt.values())
print(ed_to_origin.values())

ed_to_gt, ed_to_origin = get_avg_edit_distance(gen_res1)
print(ed_to_gt.values())
print(ed_to_origin.values())

ed_to_gt, ed_to_origin = get_avg_edit_distance(gen_res2)
print(ed_to_gt.values())
print(ed_to_origin.values())

ed_to_gt, ed_to_origin = get_avg_edit_distance(gen_res3)
print(ed_to_gt.values())
print(ed_to_origin.values())