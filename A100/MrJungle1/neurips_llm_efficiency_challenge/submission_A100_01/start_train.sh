python3 exllamav2/convert.py -i /workspace/Llama-2-70b-hf -o /workspace/exl2_tmp -c /workspace/train-00000-of-00018-60349854e9c475bb.parquet -cf /workspace/Llama-2-70b-hf-4.0bpw -b 4.0 -m  /workspace/measurement.json -nr
python3 push_to_hf.py