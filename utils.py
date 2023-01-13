import json
import pickle
from typing import Optional

import torch
import os

from transformers import DistilBertForMaskedLM, DistilBertConfig

INPUT_LENGTH = 18
TEAM_1_START_INDEX = 4
Model = DistilBertForMaskedLM
Config = DistilBertConfig
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_MODEL = 'pokemon-team-builder-transformer-v2'


def find_usage_file(format: str):
    """since some postfixes are different, this function finds the correct usage file"""
    for file in os.listdir('usage_data'):
        if format in file:
            return file
    raise Exception(f"Could not find usage file for {format}")


def make_tokens_from_team(chosen_pokemon, format):
    tokens = [0]
    tokens += ["[SEP]"]
    tokens += [format]
    tokens += ["[SEP]"]
    tokens += chosen_pokemon
    while len(tokens) < TEAM_1_START_INDEX + 6:
        tokens += ["[MASK]"]
    tokens += ["[SEP]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]"]
    assert len(tokens) == INPUT_LENGTH
    return tokens


def make_input_ids(tokens, tokenizer):
    ids = tokenizer.encode(tokens)
    ids['input_ids'] = torch.tensor(ids['input_ids']).unsqueeze(0).to(DEVICE)
    ids['attention_mask'] = torch.tensor(ids['attention_mask']).unsqueeze(0).to(DEVICE)
    return ids


def load_model(name: Optional[str] = None):
    if name is None:
        name = DEFAULT_MODEL
    with open('pickles/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    model = Model.from_pretrained(name).to(DEVICE)
    return tokenizer, model


def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)
