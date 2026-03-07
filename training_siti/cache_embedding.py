import pyarrow.parquet as pq
from typing import cast, List, Dict, Any
import re
import json
import pandas as pd
from models.ModelProviderClient import vLLMClient

INPUT_TRAIN="./data/train_extractor.parquet"
INPUT_VAL="./data/test_extractor.parquet"
EMBEDDING_MODEL_ENDPOINT="http://127.0.0.1:8001"

def is_present(value:str)->bool:
        """funzione che trova se è presente una nota oppure no"""
        if value is None:
            return False
        if isinstance(value, str) and value.strip() == '':
            return False
        return True

def create_embed(data,embedding_model):
    for task in data:
        gt = task['reward_model']['ground_truth']
        # separiamo assistant gt rispetto a extractor
        #gt_assistant_idx = [idx for idx,assistant in enumerate(gt) if assistant['role'] == 'assistant']
        #gt_extractor_idx = [idx for idx,assistant in enumerate(gt) if assistant['role'] == 'extractor']
        search_field = ['note', 'note_inefficienza']
        for field in search_field:
                task['reward_model']['ground_truth'][f'{field}_embedding'] = embedding_model.embed(prompt=gt[field], dim=256)['embeddings'][0]['embedding'] if is_present(gt[field]) == True else None
            
    return data

def main():
    # load the dataset
    train_table = pq.read_table(INPUT_TRAIN)
    train_dataset = cast(List[Dict[str, Any]], train_table.to_pylist()) 
    val_table = pq.read_table(INPUT_VAL)
    val_dataset = cast(List[Dict[str, Any]], val_table.to_pylist())


    # laod client 
    embedding_model: vLLMClient = vLLMClient(base_url=EMBEDDING_MODEL_ENDPOINT)

    train_dataset  = create_embed(train_dataset,embedding_model)
    val_dataset = create_embed(val_dataset, embedding_model)
    
    train_df = pd.DataFrame(train_dataset)
    test_df = pd.DataFrame(val_dataset)
    


    # 3 STEP: convert to parquet binary file
    train_df.to_parquet("train_extractor_w_embedding.parquet")
    test_df.to_parquet("test_extractor_w_embedding.parquet")



if __name__ == "__main__":
    main()