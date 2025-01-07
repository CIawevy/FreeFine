unset http_proxy
unset https_proxy
unset no_proxy
#python flask_mask_filter_plus.py --dir "/data/Hszhu/dataset/GRIT/" --port 8860 --subset_id 1
#python identifier_mask_filter_plus.py --dir "/data/Hszhu/dataset/GRIT/" --port 8860
python identifier_mask_filter_plus_v5_lc.py --dir /data/Hszhu/dataset/Subjects200K/ --port 8860
#python identifier_mask_filter_plus_v5.py --dir "/data/Hszhu/dataset/GRIT/" --port 7864
#python identifier_mask_filter_plus_v5.py --dir "/data/Hszhu/dataset/Subjects200K/" --port 8860
