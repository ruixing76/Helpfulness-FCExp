import os
import argparse
import numpy as np
import pandas as pd
from json_repair import repair_json
from utils.data_io import *
from utils.model import *
from utils.prompts import *
from utils.utils import *
from utils.config import *

# Define constants
MAX_LENGTH = None
BATCH_SIZE = None
EPOCHS = None
LEARNING_RATE = None
MODEL_FULL_NAME = None
MODEL_SHORT_NAME=None  # for save parameters and log
SEED = None
LANG= None
SAVE_PATH=None


# prompt without evidence
def gen_direct_prompt(row):
    return PREDICTION_DIRECT.substitute({'claim':row['claim'],'note':row['note_text']})

def gen_reason_prompt(row, reason_definitions:str):
    return PREDICTION_REASON.substitute({'claim':row['claim'],'note':row['note_text'],'reason_definitions':reason_definitions})

def gen_reason_enhanced_prompt(row):
    return PREDICTION_REASON_v3.substitute({'claim':row['claim'],'note':row['note_text']})


def gen_direct_evidence_prompt(row):
    evidence_str = row['external_link_content']['content']
    # TODO if the evidence is too long, we will use the summary
    return PREDICTION_DIRECT_EVIDENCE.substitute({'claim':row['claim'],'note':row['note_text'],'evidence':evidence_str})

def gen_reason_evidence_prompt(row):
    evidence_str = row['external_link_content']['content']
    return PREDICTION_REASON_EVIDENCE.substitute({'claim':row['claim'],'note':row['note_text'],'evidence':evidence_str})


def get_reason_definition_str(reason_def_path:str):
    with open(reason_def_path, 'r') as f:
        reason_definitions = json.load(f)
    reason_definitions_str = "\n".join([f"{key}: {value}" for key, value in reason_definitions.items()])
    return reason_definitions_str

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a multi-task model for claim notes.")
    
    # Data paths
    parser.add_argument(
        '--data_path', type=str, required=False,
        default='datapath', 
        help='Path to the data file.')
    

    parser.add_argument('--model_name', type=str, default="llama31-8b-instruct", help='Name of the LLM to use.')
    parser.add_argument('--is_flatten', action='store_true', help='Whether the data is flattened.')
    parser.add_argument('--lang', type=str, default='en', help='Language of the dataset.')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducibility.')
    parser.add_argument('--prompt_type', type=str, default='direct', help='Type of prompt to use.')
    parser.add_argument('--reason_def_path', type=str, default='./data/seed_def.json', help='Path to the reason definition file.')
    
    return parser.parse_args()

def main():
    global MAX_LENGTH
    global BATCH_SIZE
    global EPOCHS
    global LEARNING_RATE
    global MODEL_FULL_NAME
    global MODEL_SHORT_NAME
    global SEED
    global LANG
    global SAVE_PATH

    args = parse_arguments()

    MODEL_FULL_NAME = args.model_name
    MODEL_SHORT_NAME = args.model_name.split("/")[-1].strip()
    SEED = args.seed
    LANG = args.lang
    SAVE_PATH = f"./predict/{LANG}/{MODEL_SHORT_NAME}"


    print("Predicting...")
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Set random seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Model inference
    print("Loading the dataset...")
    # remove data if there's no summary key
    test_data=read_jsonl(args.data_path)
    if not args.is_flatten:
        print("Flattening the data...")
        test_data=flatten_data(test_data)
    # if external_link_content is None, put the content as empty string
    # if external_link_content is not None, but content or summary is None, put the content or summary as empty string
    for each in test_data:
        if 'external_link_content' not in each or each['external_link_content'] is None:
            each['external_link_content'] = {'content': '', 'summary': ''}
        else:
            if each['external_link_content'].get('content') is None:
                each['external_link_content']['content'] = ''
            if each['external_link_content'].get('summary') is None:
                each['external_link_content']['summary'] = ''
    data_df=pd.DataFrame.from_dict(test_data)
    # data_df=data_df[:10]
   
    if args.prompt_type == 'direct':
        print("Generating direct prompts...")
        prompts=data_df.apply(lambda row:gen_direct_prompt(row),axis=1).to_list()
    elif args.prompt_type == 'definition':
        print("Generating definition prompts...")
        reason_definitions_str=get_reason_definition_str(args.reason_def_path)
        prompts=data_df.apply(lambda row:gen_reason_prompt(row, reason_definitions_str),axis=1).to_list()
    elif args.prompt_type == 'enhanced':
        print("Generating enhanced prompts...")
        prompts=data_df.apply(lambda row:gen_reason_enhanced_prompt(row),axis=1).to_list()
    else:
        raise ValueError(f"Invalid prompt type: {args.prompt_type}")

    if MODEL_FULL_NAME.startswith('gpt') or MODEL_FULL_NAME.startswith('ft:gpt'):
        generator = OpenAIGenerator(api_key=API_KEY, model=MODEL_FULL_NAME, is_batch=True)
        results = generator.generate(prompts, max_output_length=512,
                                     system_prompt=SYSTEM_ROLE)
    else:
        if MODEL_FULL_NAME in MODEL_mapping:
            model_name = MODEL_mapping[MODEL_FULL_NAME]
        else:
            model_name = MODEL_FULL_NAME
        generator = Generator(model_name=model_name, tensor_parallel_size=1)
        generator.load_model()
        results = generator.generate(prompts, max_output_length=256, system_prompt=SYSTEM_ROLE)
    # clean response, if the output is directly JSON
    results=[repair_json(each) for each in results]
    # otherwise we need to process the output and turn it into JSON
    # ...
    data_df['model_predict']=results
    data_path=f"{SAVE_PATH}/{MODEL_SHORT_NAME}_prediction.jsonl"
    data_df.to_json(data_path, orient="records", lines=True)


def test_prompt():
    test_data_path = 'datapath'
    reason_def_path = 'datapath'
    # read gen_def data and generate prompt
    data_df=pd.DataFrame.from_dict(read_jsonl(test_data_path))
    reason_definitions_str=get_reason_definition_str(reason_def_path)
    prompts=data_df.apply(lambda row:gen_reason_prompt(row, reason_definitions_str),axis=1).to_list()
    print(prompts[0])


if __name__ == "__main__":
    main()