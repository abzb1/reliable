import json

def get_json_fail_rate(gen_res):
    ks = ['original', 'half_cut_0', 'half_cut_1', 'half_cut_2', 'half_cut_3', 'third_cut_0', 'third_cut_1', 'third_cut_2', 'third_cut_3', 'quarter_cut_0', 'quarter_cut_1', 'quarter_cut_2', 'quarter_cut_3']

    json_failed = {key: 0 for key in ks}
    json_success = {key: 0 for key in ks}
    for key in ks:

        for q_idx in range(len(gen_res)):
            result = gen_res[q_idx][f"{q_idx:03d}_{key}"]
            
            if result["is_json_structured"]:
                json_success[key] += 1
            else:
                json_failed[key] += 1

    json_failed = {k: v/len(gen_res) for k, v in json_failed.items()}
    json_success = {k: v/len(gen_res) for k, v in json_success.items()}

    return json_failed, json_success

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

json_failed, json_success = get_json_fail_rate(hcx_res)
print("json_failed", json_failed.values())
print("json_success", json_success.values())

json_failed, json_success = get_json_fail_rate(qwenvl_res)
print("json_failed", json_failed.values())
print("json_success", json_success.values())

json_failed, json_success = get_json_fail_rate(qwenvl_trained_res)
print("json_failed", json_failed.values())
print("json_success", json_success.values())

json_failed, json_success = get_json_fail_rate(gen_res1)
print("json_failed", json_failed.values())
print("json_success", json_success.values())

json_failed, json_success = get_json_fail_rate(gen_res2)
print("json_failed", json_failed.values())
print("json_success", json_success.values())

json_failed, json_success = get_json_fail_rate(gen_res3)
print("json_failed", json_failed.values())
print("json_success", json_success.values())