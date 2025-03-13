import os
import datasets
import argparse
import json
import requests
from tqdm import tqdm
import zipfile
import random
import sys
# 将 /data/yu12345/Agent-R1/ 添加到 sys.path
sys.path.append('/data/yu12345/Agent-R1/')
from verl.utils.hdfs_io import copy, makedirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/crag')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', type=int, default=1280,
                        help='Number of training samples to use')
    parser.add_argument('--val_size', type=int, default=128,
                        help='Number of validation samples to use')
    args = parser.parse_args()
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    #完成下载
    train_file = os.path.join(local_dir, 'task1_split_0_no_link.json')
    dev_file = os.path.join(local_dir, 'task1_split_1_no_link.json')
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
        
    with open(dev_file, 'r', encoding='utf-8') as f:
        validation_data = json.load(f)
        
    # Inspect the structure of the first item to understand the data format
    print("Sample data structure:", json.dumps(train_data[0], indent=2)[:500] + "...")
        
    train_dataset = datasets.Dataset.from_dict({
            'interaction_id': [item['interaction_id'] for item in train_data],
            'query_time': [item['query_time'] for item in train_data],
            'domain': [str(item.get('domain', '')) for item in train_data],
            'question_type': [str(item.get('question_type', '')) for item in train_data],
            'static_or_dynamic': [str(item.get('static_or_dynamic', '')) for item in train_data],
            'query': [str(item.get('query', '')) for item in train_data],
            'answer': [str(item.get('answer', '')) for item in train_data],
            "search_results": [str(item.get('search_results', '')) for item in train_data]
            
        })
    
    validation_dataset = datasets.Dataset.from_dict({
            'interaction_id': [item['interaction_id'] for item in validation_data],
            'query_time': [item['query_time'] for item in validation_data],
            'domain': [str(item.get('domain', '')) for item in validation_data],
            'question_type': [str(item.get('question_type', '')) for item in validation_data],
            'static_or_dynamic': [str(item.get('static_or_dynamic', '')) for item in validation_data],
            'query': [str(item.get('query', '')) for item in validation_data],
            'answer': [str(item.get('answer', '')) for item in validation_data],
            "search_results": [str(item.get('search_results', '')) for item in validation_data]
        })

    if args.train_size is not None:
            indices = random.sample(range(len(train_dataset)), args.train_size)
            train_dataset = train_dataset.select(indices)
    if args.val_size is not None:
            indices = random.sample(range(len(validation_dataset)), args.val_size)
            validation_dataset = validation_dataset.select(indices)
            
    instruction_following = """Answer the given question. You can use the tools provided to you to answer the question. You can use the tool as many times as you want.
You must first conduct reasoning inside <think>...</think>. If you need to use the tool, you can use the tool call <tool_call>...</tool_call> to call the tool after <think>...</think>.
When you have the final answer, you can output the answer inside <answer>...</answer>.

Output format for tool call:
<think>
...
</think>
<tool_call>
...
</tool_call>

Output format for answer:
<think>
...
</think>
<answer>
...
</answer>
"""        

    # Process each data item
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop('query')
            question = instruction_following + "Question: " + question_raw
            
            answer_raw = example.pop('answer')
            
            # Parse the supporting facts from JSON string back to Python object if needed
            search_results_str = example.get('search_results', '[]')
            try:
                search_results = json.loads(search_results_str)
            except (json.JSONDecodeError, TypeError):
                search_results = []
            
            # Convert all data to string format to avoid type issues
           
            data = {
                "data_source": "crag",
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "crag",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer_raw
                },
                "extra_info": {
                    'split': split,
                    'index': str(idx),
                    'answer': answer_raw,
                    'question': question_raw,
                    'search_results': json.dumps(search_results),  # Store as JSON string
                    'interaction_id': str(example.get('interaction_id', '')),
                    'query_time': str(example.get('query_time', '')),
                    'domain': str(example.get('domain', '')),
                    'question_type': str(example.get('question_type', '')),
                    'static_or_dynamic': str(example.get('static_or_dynamic', ''))
                }
            }
            return data

        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    validation_dataset = validation_dataset.map(function=make_map_fn('validation'), with_indices=True)
    
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    validation_dataset.to_parquet(os.path.join(local_dir, 'validation.parquet'))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)