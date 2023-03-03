from typing import Union, List

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


class BERTBattleTokenizer(BattleTokenizer):
    def __init__(self, special_token_map=None, formats=None):

        super().__init__(special_token_map, formats)

        self.name = "bert_tokenizer"

    def encode(self, tokens: Union[List[Union[str, int]], str, int], ignore_assert: bool = False):
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
