from typing import Union, List

import numpy as np
from tqdm import tqdm

from data_classes import Battle
from replay_parser import parse_replay
from utils import *


class BattleTokenizer:
    def __init__(self, special_token_map=None, formats=None):
        if special_token_map is None:
            self.special_token_map = {
                'P1 Win': 0,
                'P2 Win': 1,
                "[PAD]": 2,
                "[SEP]": 3,
                '[MASK]': 4,
            }
            self.special_tokens = ['P1 Win', 'P2 Win', "[PAD]", "[SEP]", '[MASK]']
        else:
            self.special_token_map = special_token_map
            self.special_tokens = list(special_token_map.keys())
        if formats is None:
            self.formats = [
                'gen9ou',
                'gen9monotype',
                'gen9doublesou',
                'gen9vgc2023series1',
                'gen9nationaldex',
                'gen9uu',
                'gen9ru',
                'gen9ubers',
                'gen9vgc2023series2'
            ]
        else:
            self.formats = formats

        self.name = "tokenizer"
        self.token_map = self.special_token_map
        for format in self.formats:
            self.token_map[format] = len(self.token_map)
        self.vocab_size = len(self.token_map)
        self.rating_dist = {}
        self.expected_input_length = INPUT_LENGTH

    def get_vocab(self):
        return list(self.token_map.keys())

    def from_file(self, format, bid):
        """
        encode a battle into a list of ids
        :param format: format of the battle
        :param bid: battle id
        :return: list of ids
        """
        tokens = []
        battle: Battle = parse_replay(format, bid)
        if battle is None:
            return None
        winner_index = 0 if battle.winner == battle.p1.name else 1
        tokens.append(winner_index)
        tokens.append("[SEP]")
        tokens.append(battle.format)
        tokens.append("[SEP]")
        for i, player in enumerate([battle.p1, battle.p2]):
            # tokens.append(player.rating)
            for pokemon in player.team:
                tokens.append(pokemon.name)
            tokens.append("[SEP]")

        # keep rating distribution
        if battle.rated:
            bucket = battle.rating
        else:
            bucket = "unrated"
        if bucket not in self.rating_dist:
            self.rating_dist[bucket] = 0
        self.rating_dist[bucket] += 1
        return self.encode(tokens)

    def _encode_token(self, token: Union[str, int]):
        if isinstance(token, int):
            return token
        elif isinstance(token, str):
            if token == "NaP":
                return self.special_token_map["[PAD]"]
            if token not in self.token_map:
                self.token_map[token] = len(self.token_map)
                print(f"New pokemon: {token} -> {self.token_map[token]}")
            return self.token_map[token]

    def _encode_list(self, tokens: List[Union[str, int]]):
        return [self._encode_token(token) for token in tokens]

    def encode(self, tokens: Union[List[Union[str, int]], str, int], ignore_assert: bool = False):
        """
        encode a list of tokens into a list of ids
        :param tokens:
        :param ignore_assert:
        :return:
        """
        if isinstance(tokens, int) or isinstance(tokens, str):
            return self._encode_token(tokens)
        ids = []
        for token in tokens[:]:
            ids.append(self._encode_token(token))

        while len(ids) < self.expected_input_length:
            ids.append(self.special_token_map["[PAD]"])

        if not ignore_assert:
            assert ids[0] in self._encode_list([0, 1, '[MASK]'])
            assert ids[2] in self._encode_list(self.formats + ['[MASK]'])

        attention_mask = [1 if i != self.special_token_map["[PAD]"] else 0 for i in ids[:]]
        ids = {"input_ids": ids[:], "attention_mask": attention_mask}
        self.vocab_size = len(self.token_map)
        return ids

    def decode(self, pokemon_id: Union[int, List[int]]):
        """
        decode a (list of) id(s) into a (list of) token(s)
        :param pokemon_id: list of ids or single id
        :return: list of tokens or single token
        """
        reverse_dict = {v: k for k, v in self.token_map.items()}
        if isinstance(pokemon_id, int):
            try:
                return reverse_dict[pokemon_id]
            except KeyError as e:
                # print(f"KeyError: {pokemon_id}")
                return "NaP"
        elif isinstance(pokemon_id, list):
            return [self.decode(i) for i in pokemon_id]
        try:
            decoded = reverse_dict[int(pokemon_id)]
            return decoded
        except:
            raise ValueError(f"pokemon_id must be int or list, not {type(pokemon_id)}, {pokemon_id}")

    def shuffle_teams_dataset(self, dataset):
        """
        Shuffles the teams in a dataset, keep ids on the same team
        :param self: tokenizer to use
        :param dataset: tokenized dataset to shuffle
        :return: shuffled dataset
        """
        new_dataset = []
        for data in dataset:
            inputs = data['input_ids'][:]
            if DISTILBERT_LIKE:
                team1, team2 = find_teams_distilbert(data, tokenizer=self)
                np.random.shuffle(team1)
                np.random.shuffle(team2)
                new_input = inputs[:TEAM_1_START_INDEX] + \
                            team1 + \
                            [self.token_map['[SEP]']] + \
                            team2 + \
                            [self.token_map['[SEP]']]
                while len(new_input) < INPUT_LENGTH:
                    new_input.append(self.special_token_map['[PAD]'])
                new_dataset.append({'input_ids': new_input,
                                    'attention_mask': data['attention_mask'][:]
                                    })
            else:
                team1, team2 = find_teams_bert(data)
                np.random.shuffle(team1)
                np.random.shuffle(team2)
                new_input = inputs[:TEAM_1_START_INDEX] + \
                            team1 + \
                            team2
                while len(new_input) < INPUT_LENGTH:
                    new_input.append(self.special_token_map['[PAD]'])
                new_dataset.append({'input_ids': new_input,
                                    'attention_mask': data['attention_mask'][:],
                                    'token_type_ids': data['token_type_ids'][:],
                                    'position_ids': data['position_ids'][:]
                                    })
            # assert that padding is the same
            assert inputs.count(self.token_map['[PAD]']) == new_dataset[-1]['input_ids'].count(
                self.token_map['[PAD]'])
            assert max(new_dataset[-1]['input_ids']) < self.vocab_size

        return new_dataset

    def switched_sides_dataset(self, dataset):
        """
        Switches the sides of the teams in a dataset, keep ids on the same team
        :param self: tokenizer to use
        :param dataset: tokenized dataset to switch
        :return: switched dataset
        """
        switched_dataset = []
        for data in dataset:
            inputs = data['input_ids'][:]
            inputs[0] = 1 if inputs[0] == 0 else 0
            if DISTILBERT_LIKE:
                team1, team2 = find_teams_distilbert(data, tokenizer=self)
                new_inputs = inputs[:TEAM_1_START_INDEX] + \
                             team2 + \
                             [self.token_map['[SEP]']] + \
                             team1 + \
                             [self.token_map['[SEP]']]
                while len(new_inputs) < INPUT_LENGTH:
                    new_inputs.append(self.special_token_map['[PAD]'])

                switched_dataset.append({
                    'input_ids': new_inputs,
                    'attention_mask': data['attention_mask'][:]
                })
            else:
                team1, team2 = find_teams_bert(data)
                new_inputs = inputs[:TEAM_1_START_INDEX] + \
                             team2 + \
                             team1
                while len(new_inputs) < INPUT_LENGTH:
                    new_inputs.append(self.token_map['[PAD]'])

                new_types = data['token_type_ids'][:TEAM_1_START_INDEX] + \
                            [2 for _ in range(len(team2))] + \
                            [3 for _ in range(len(team1))]
                while len(new_types) < INPUT_LENGTH:
                    new_types.append(max(new_types))
                assert max(new_types) == 3

                switched_dataset.append({
                    'input_ids': new_inputs,
                    'attention_mask': data['attention_mask'][:],
                    'token_type_ids': new_types,
                    'position_ids': data['position_ids'][:]
                })

            # assert that padding is the same
            assert inputs.count(self.token_map['[PAD]']) == switched_dataset[-1]['input_ids'].count(
                self.token_map['[PAD]'])
            assert data['input_ids'][0] != switched_dataset[-1]['input_ids'][0]

        # for i, data in enumerate(switched_dataset):
        #     print(f"Original {i}: {dataset[i]}")
        #     print(f"Switched {i}: {data}")
        return switched_dataset


class BERTBattleTokenizer(BattleTokenizer):
    def __init__(self, special_token_map=None, formats=None):

        super().__init__(special_token_map, formats)

        self.name = "bert_tokenizer"

    def encode(self, tokens: Union[List[Union[str, int]], str, int], ignore_assert: bool = False):
        """
        encode a list of tokens into a list of ids
        :param tokens: list of tokens
        :param ignore_assert: ignore assertion error
        :return: list of ids
        """
        if isinstance(tokens, int) or isinstance(tokens, str):
            return self._encode_token(tokens)
        ids = []
        type_ids = [0] + [1]
        current_type = 2
        for i, token in enumerate(tokens[:]):
            if token == "[SEP]":
                current_type += 1
                continue
            ids.append(self._encode_token(token))
            if i >= 2:
                type_ids.append(current_type)

        while len(ids) < self.expected_input_length:
            ids.append(self.special_token_map["[PAD]"])
            type_ids.append(max(type_ids))

        attention_mask = [1 if i != self.special_token_map["[PAD]"] else 0 for i in ids[:]]
        position_ids = [0 for i in range(len(ids))]

        # print(f"ids: {ids}")
        if not ignore_assert:
            assert len(ids) == len(type_ids) == len(attention_mask) == len(position_ids)
            assert len(ids) == self.expected_input_length
            assert ids[0] in self._encode_list([0, 1, '[MASK]'])
            assert ids[1] in self._encode_list(self.formats + ['[MASK]'])
            assert max(type_ids) == 3

        ids = {
            "input_ids": ids[:],
            "attention_mask": attention_mask,
            "token_type_ids": type_ids,
            "position_ids": position_ids
        }
        self.vocab_size = len(self.token_map)
        return ids

    def from_file(self, format, bid):
        """
        encode a battle into a list of ids
        :param format: format of the battle
        :param bid: battle id
        :return: list of ids
        """
        tokens = []
        battle: Battle = parse_replay(format, bid)
        if battle is None:
            return None
        winner_index = 'P1 Win' if battle.winner == battle.p1.name else 'P2 Win'
        tokens.append(winner_index)
        tokens.append(battle.format)
        for i, player in enumerate([battle.p1, battle.p2]):
            # tokens.append(player.rating)
            for pokemon in player.team:
                tokens.append(pokemon.name)
            tokens.append("[SEP]")
        if len(tokens) > 16:
            print(f"tokens: {tokens}")
        return self.encode(tokens)


def tokenize_dataset(formats: List[str], name: str):
    """
    Tokenizes the dataset
    :param formats: list of formats to tokenize
    :param name: name of the dataset
    :return: tokenized dataset
    """
    # remove training step from name
    for x in ["-pretrain", "-team-builder", "-viability", "-winner"]:
        name = name.replace(x, "")

    tokenizer_file = f"pickles/tokenizer.pkl" if DISTILBERT_LIKE else f"pickles/bert_tokenizer-{name}.pkl"
    if not os.path.exists(tokenizer_file):
        tokenizer = BattleTokenizer() if DISTILBERT_LIKE else BERTBattleTokenizer(formats=formats)
    else:
        with open(tokenizer_file, 'rb') as f:
            tokenizer = pickle.load(f)

    if not os.path.exists(f'pickles/dataset-{name}.pkl') or not os.path.exists(tokenizer_file):
        tokenizer.rating_dist = {}
        dataset = []
        for format in tqdm(formats, desc=f"Tokenizing formats", leave=False):
            # check if directory exists
            if not os.path.exists(f'replays/{format}'):
                continue
            for filename in tqdm(os.listdir(f'replays/{format}'), desc=f"Tokenizing {format} replays",
                                 leave=False):
                if filename.endswith('.log'):
                    f = os.path.join(f'replays/{format}', filename)
                    # checking if it is a file
                    if os.path.isfile(f):
                        id = filename[:-4]
                        tokens = tokenizer.from_file(format, id)
                        # print(tokens)
                        if tokens is not None:
                            dataset.append(tokens)
        with open(tokenizer_file, 'wb') as f:
            pickle.dump(tokenizer, f)
        with open(f'pickles/dataset-{name}.pkl', 'wb') as f:
            pickle.dump(dataset, f)

    else:
        with open(f'pickles/dataset-{name}.pkl', 'rb') as f:
            dataset = pickle.load(f)
    return dataset, tokenizer