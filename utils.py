import json
import pickle
from typing import Optional

import torch
import os

from transformers import DistilBertForMaskedLM, DistilBertConfig, DebertaConfig, DebertaForMaskedLM, RobertaConfig, RobertaForMaskedLM

Model = DebertaForMaskedLM
Config = DebertaConfig
DISTILBERT_LIKE = False
# DISTILBERT_LIKE = True
# Model = DistilBertForMaskedLM
# Config = DistilBertConfig
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_MODEL = 'pokemon-team-builder-transformer-deberta4-large'
INPUT_LENGTH = 18 if DISTILBERT_LIKE else 14
TEAM_1_START_INDEX = 4 if DISTILBERT_LIKE else 2


def find_usage_file(format: str):
    """since some postfixes are different, this function finds the correct usage file"""
    for file in os.listdir('usage_data'):
        if format in file:
            return file
    raise FileNotFoundError(f"Could not find usage file for {format}")


def make_tokens_from_team(chosen_pokemon, format):
    tokens = [0]
    if DISTILBERT_LIKE:
        tokens += ["[SEP]"]
        tokens += [format]
        tokens += ["[SEP]"]
        tokens += chosen_pokemon
        tokens += ["[MASK]"]
        while len(tokens) < TEAM_1_START_INDEX + 6:
            tokens += ["[PAD]"]
        tokens += ["[SEP]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]"]
    else:
        tokens += [format]
        tokens += chosen_pokemon
        tokens += ["[MASK]"]
        while len(tokens) < TEAM_1_START_INDEX + 6:
            tokens += ["[PAD]"]
        tokens += ["[SEP]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]"]
    return tokens


def make_input_ids(tokens, tokenizer):
    ids = tokenizer.encode(tokens)
    assert len(ids['input_ids']) == INPUT_LENGTH
    ids['input_ids'] = torch.tensor(ids['input_ids']).unsqueeze(0).to(DEVICE)
    ids['attention_mask'] = torch.tensor(ids['attention_mask']).unsqueeze(0).to(DEVICE)
    if 'position_ids' in ids:
        ids['position_ids'] = torch.tensor(ids['position_ids']).unsqueeze(0).to(DEVICE)
    if 'token_type_ids' in ids:
        ids['token_type_ids'] = torch.tensor(ids['token_type_ids']).unsqueeze(0).to(DEVICE)
    return ids


def load_model(name: Optional[str] = None):
    if name is None:
        name = DEFAULT_MODEL
    pickle_file = f'pickles/tokenizer.pkl' if DISTILBERT_LIKE else f'pickles/bert_tokenizer.pkl'
    with open(pickle_file, 'rb') as f:
        tokenizer = pickle.load(f)
    model = Model.from_pretrained(name).to(DEVICE)
    return tokenizer, model


def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)
