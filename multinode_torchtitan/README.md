```bash
git clone https://github.com/pytorch/torchtitan
cd torchtitan
pip install -r requirements.txt
python scripts/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original"
huggingface-cli download allenai/c4 --repo-type dataset # If used
```

1. slurm script inside torchtitan folder
2. 
`./torchtitan/models/llama3/train_configs/llama3_8b.toml` modified
- Using c4_test instead of c4