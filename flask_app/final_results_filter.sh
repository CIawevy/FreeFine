unset http_proxy
unset https_proxy
unset no_proxy
#python flask_mask_filter_plus.py --dir "/data/Hszhu/dataset/GRIT/" --port 8860 --subset_id 1
#python identifier_mask_filter_plus.py --dir "/data/Hszhu/dataset/GRIT/" --port 8860
python identifier_final_filter.py --dir "/data/Hszhu/dataset/PIE-Bench_v1/" --port 7864

