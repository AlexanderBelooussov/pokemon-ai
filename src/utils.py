import json
import pickle
from typing import Optional

import torch
import os

from transformers import DistilBertForMaskedLM, DistilBertConfig, DebertaConfig, DebertaForMaskedLM, RobertaConfig, \
    RobertaForMaskedLM

Model = DebertaForMaskedLM
Config = DebertaConfig
DISTILBERT_LIKE = False
# DISTILBERT_LIKE = True
# Model = DistilBertForMaskedLM
# Config = DistilBertConfig
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_MODEL = 'pokemon-team-builder-transformer-deberta6-team-builder'
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
        if len(chosen_pokemon) < 6:
            tokens += ["[MASK]"]
        while len(tokens) < TEAM_1_START_INDEX + 6:
            tokens += ["[MASK]"]
        tokens += ["[SEP]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[SEP]"]
    else:
        tokens += [format]
        tokens += chosen_pokemon
        if len(chosen_pokemon) < 6:
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


def make_ids_from_team(chosen_pokemon, format, tokenizer):
    tokens = make_tokens_from_team(chosen_pokemon, format)
    return make_input_ids(tokens, tokenizer)


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


def restore_ids(inputs):
    """
    Restore ids to the original ids
    This is necessary when inputs are changed when masking
    This resets the ids so they can be remasked
    :param inputs: inputs to restore
    :return: restored inputs
    """
    if 'labels' not in inputs:
        return inputs
    for j, label in enumerate(inputs['labels']):
        if label != -100:
            inputs['input_ids'][j] = label
    inputs.pop('labels')
    return inputs


def find_teams_distilbert(inputs, tokenizer):
    inputs = inputs['input_ids'][:]
    team1 = []
    sep_index = 0
    team2 = []
    for i, token in enumerate(inputs):
        if token == tokenizer.special_token_map['[SEP]']:
            if len(team1) == 0:
                team1 = inputs[TEAM_1_START_INDEX:i]
                sep_index = i
                continue
            elif len(team2) == 0:
                team2 = inputs[sep_index + 1:i]
                break
    return team1, team2


def find_teams_bert(inputs):
    ids = inputs['input_ids'][:]
    types = inputs['token_type_ids'][:]
    team1 = []
    team2 = []
    for i, token in enumerate(ids):
        if types[i] == 2:
            team1.append(token)
        elif types[i] == 3:
            team2.append(token)
    return team1, team2