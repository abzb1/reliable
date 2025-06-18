python result_ed_mp.py \
--pairs \
"/data1/home/ohs/workspace/rai/original/generated_jsons/hcx":"/data1/home/ohs/workspace/rai/whole_result/hcx_generation_res.jsonl" \
"/data1/home/ohs/workspace/rai/original/generated_jsons/qwenvl":"/data1/home/ohs/workspace/rai/whole_result/qwenvl_generation_res.jsonl" \
"/data1/home/ohs/workspace/rai/trained/only/generated_jsons/qwenvl":"/data1/home/ohs/workspace/rai/whole_result/qwenvl_train_only_save_fname.jsonl" \
"/data1/home/ohs/workspace/rai/trained/rotated_1/generated_jsons/qwenvl":"/data1/home/ohs/workspace/rai/whole_result/qwenvl_train_1epoch_generation_res.jsonl" \
"/data1/home/ohs/workspace/rai/trained/rotated_2/generated_jsons/qwenvl":"/data1/home/ohs/workspace/rai/whole_result/qwenvl_train_2epoch_generation_res.jsonl" \
"/data1/home/ohs/workspace/rai/trained/rotated_3/generated_jsons/qwenvl":"/data1/home/ohs/workspace/rai/whole_result/qwenvl_train_3epoch_generation_res.jsonl" \
--num-workers 20 \
--split test