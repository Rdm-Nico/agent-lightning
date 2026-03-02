import pyarrow.parquet as pq
from typing import cast, List, Dict, Any
import re
import json
import pandas as pd
from models.ModelProviderClient import vLLMClient

INPUT_TRAIN="./data/train.parquet"
INPUT_VAL="./data/test.parquet"
EMBEDDING_MODEL_ENDPOINT="http://127.0.0.1:8001"


def create_embed(data,embedding_model):
    for task in data:
        gt = task['reward_model']['ground_truth']
        # separiamo assistant gt rispetto a extractor
        gt_assistant_idx = [idx for idx,assistant in enumerate(gt) if assistant['role'] == 'assistant']

        for idx in gt_assistant_idx:

            # prendiamo la gt
            clean_gt_str =  re.sub(r"<TOOLCALL>\[(.*?)\]</TOOLCALL>\n?",r"\1", gt[idx]['content'])
            clean_gt = json.loads(clean_gt_str)
            # se siamo nel push andiamo avanti
            if "push" in clean_gt['arguments']:
                continue
            
            task['reward_model']['ground_truth'][idx]["embedding"] = embedding_model.embed(clean_gt['arguments']['summary'], dim=256)['embeddings'][0]['embedding']
    
    return data

def main():
    # load the dataset
    train_table = pq.read_table(INPUT_TRAIN)
    train_dataset = cast(List[Dict[str, Any]], train_table.to_pylist()) 
    val_table = pq.read_table(INPUT_VAL)
    val_dataset = cast(List[Dict[str, Any]], val_table.to_pylist())


    # laod client 
    embedding_model: vLLMClient = vLLMClient(base_url=EMBEDDING_MODEL_ENDPOINT)
    i = 0
    train_dataset  = create_embed(train_dataset,embedding_model)
    val_dataset = create_embed(val_dataset, embedding_model)
    
    
    train_df = pd.DataFrame(train_dataset)
    test_df = pd.DataFrame(val_dataset)
    
    print(train_df.head(3))

    # 3 STEP: convert to parquet binary file
    #train_df.to_parquet("train_extractor.parquet")
    #test_df.to_parquet("test_extractor.parquet")



if __name__ == "__main__":
    main()