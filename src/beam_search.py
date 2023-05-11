# finding best team with beam search
import warnings

import numpy as np
import torch
from tqdm import tqdm

from utils import make_tokens_from_team, make_input_ids, TEAM_1_START_INDEX, load_model


def filter_beams(new_candidates, i, n_beams=5, silent=False):
    """
    only keep n_beams candidates
    remove duplicate teams
    remove teams with duplicates
    :param new_candidates: Candidates to filter, list of tuples (team, prob)
    :param i: Step of beam search
    :param n_beams: Number of beams to keep
    :param silent: Whether to mute tqdm progress bars
    :return: dict of candidates -> probability
    """
    to_keep = {}
    seen = []
    new_candidates = sorted(new_candidates.items(), key=lambda x: x[1], reverse=True)
    for candidate, c_prob in tqdm(new_candidates, desc="postprocessing", leave=False, disable=silent):
        # print(candidate)
        sorted_candidate = tuple(sorted(candidate))
        if sorted_candidate not in seen and len(set(candidate)) == i + 1:
            to_keep[candidate] = c_prob
            seen.append(sorted_candidate)
        if len(to_keep) == n_beams:
            break
    return to_keep


def candidates_from_logits(tokenizer, logits, n_beams=5, chosen_pokemon=None, prob=0, forbidden_pokemon=None):
    """

    :param tokenizer: Tokenizer to decode ids to Pokémon names
    :param mask_logits:
    :param n_beams: Number of beams to keep
    :param chosen_pokemon: Pokémon already chosen
    :param prob: Base probability of already chosen Pokémon
    :param forbidden_pokemon: List of Pokémon to not include in candidates
    :return: dict of candidates -> probability
    """
    if chosen_pokemon is None:
        chosen_pokemon = []
    if forbidden_pokemon is None:
        forbidden_pokemon = []
    candidates = {}
    k = n_beams + len(forbidden_pokemon)
    k = min(k, logits.shape[0])
    top_k = torch.topk(logits, k)
    # limit top k to be at most the number of tokens
    range_lim = min(n_beams + len(forbidden_pokemon), logits.shape[0])
    for i in range(range_lim):
        candidate = chosen_pokemon[:]
        poke = tokenizer.decode(int(top_k.indices[i]))
        if poke in forbidden_pokemon:
            continue
        candidate.append(poke)
        value = np.log(float(top_k.values[i])) + prob
        candidates[tuple(candidate)] = value
    return candidates


def beam_step_from_batch(logits, tokenizer, chosen_pokemon, prob, n_beams=5, forbidden_pokemon=None):
    """
    Perform beam step from logits
    :param logits: Logits from model
    :param tokenizer: Tokenizer to decode ids to Pokémon names
    :param chosen_pokemon: Pokémon already chosen
    :param prob: Base probability of already chosen Pokémon
    :param n_beams: Number of beams to keep
    :param forbidden_pokemon: List of Pokémon to not include in candidates
    :return: dict of candidates -> probability
    """
    if forbidden_pokemon is None:
        forbidden_pokemon = []
    logits = torch.softmax(logits, dim=0)
    return candidates_from_logits(tokenizer, logits, n_beams=n_beams, chosen_pokemon=chosen_pokemon, prob=prob,
                                  forbidden_pokemon=forbidden_pokemon)


def beam_search_batched(model=None, tokenizer=None, chosen_pokemon=None, format='gen9ou', n_beams=5,
                        forbidden_pokemon=None, batch_size=200, silent=False, total_steps=6):
    """
    Perform beam search, batched for speed
    :param model: Team builder model
    :param tokenizer: Tokenizer to decode ids to Pokémon names
    :param chosen_pokemon: Pokémon already chosen
    :param format: Format to build team for
    :param n_beams: Number of beams to keep
    :param forbidden_pokemon: List of Pokémon to not include in candidates
    :param batch_size: Batch size for beam search
    :param silent: Whether to mute tqdm progress bars
    :param total_steps: Total number of steps to perform beam search for, should be 6 - len(chosen_pokemon)
    :return: list of tuples (team, prob)
    """
    total_steps = min(total_steps, 6)
    if model is None:
        tokenizer, model = load_model()
    if tokenizer is None:
        tokenizer, _ = load_model()
    if chosen_pokemon is None:
        chosen_pokemon = []
    if forbidden_pokemon is None:
        forbidden_pokemon = []
    t = len(chosen_pokemon)
    candidates = {tuple(chosen_pokemon): 1}
    model.eval()
    for i in tqdm(range(t, total_steps), desc="beam search", leave=True, disable=silent):
        new_candidates = {}
        batch_dataset = []
        for candidate in tqdm(candidates.keys(), desc="making data", leave=False, disable=silent):
            tokens = make_tokens_from_team(candidate[:], format)
            ids = make_input_ids(tokens, tokenizer)
            batch_dataset.append(ids)
        for k in tqdm(range(0, len(batch_dataset), batch_size), desc="beam step", leave=False, disable=silent):
            batch = batch_dataset[k:k + batch_size]
            input_ids = torch.cat([x["input_ids"] for x in batch], dim=0)
            attention_mask = torch.cat([x["attention_mask"] for x in batch], dim=0)
            if "token_type_ids" in batch[0]:
                token_type_ids = torch.cat([x["token_type_ids"] for x in batch], dim=0)
                position_ids = torch.cat([x["position_ids"] for x in batch], dim=0)
                outputs = model(**{
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                    "position_ids": position_ids})
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            for j, c in tqdm(enumerate(list(candidates.keys())[k:k + batch_size]), desc="adding candidates",
                             leave=False, total=len(batch), disable=silent):
                logits_j = logits[j, len(c) + TEAM_1_START_INDEX, :]
                new_candidates.update(beam_step_from_batch(
                    logits_j,
                    tokenizer,
                    list(c),
                    candidates[c],
                    n_beams=n_beams + 1,
                    forbidden_pokemon=forbidden_pokemon
                ))

        candidates = filter_beams(new_candidates, i, n_beams=n_beams, silent=silent)
        if len(candidates) < n_beams and not silent:
            warnings.warn(f"Could not find {n_beams} candidates at step {i + 1}, only found {len(candidates)}")
    # return best candidate without a duplicate
    return sorted(candidates.items(), key=lambda x: x[1], reverse=True)


def beam_step(model, tokenizer, chosen_pokemon, format, prob, n_beams=5, forbidden_pokemon=None):
    if forbidden_pokemon is None:
        forbidden_pokemon = []
    tokens = make_tokens_from_team(chosen_pokemon[:], format)
    ids = make_input_ids(tokens, tokenizer)
    logits = model(**ids).logits
    mask_logits = logits[0, len(chosen_pokemon) + TEAM_1_START_INDEX, :]
    mask_logits = torch.softmax(mask_logits, dim=0)
    return candidates_from_logits(
        tokenizer,
        mask_logits,
        n_beams=n_beams,
        chosen_pokemon=chosen_pokemon,
        prob=prob,
        forbidden_pokemon=forbidden_pokemon
    )


def beam_search(model=None, tokenizer=None, chosen_pokemon=None, format='gen9ou', n_beams=5, forbidden_pokemon=None):
    if model is None:
        tokenizer, model = load_model()
    if tokenizer is None:
        tokenizer, _ = load_model()
    if chosen_pokemon is None:
        chosen_pokemon = []
    if forbidden_pokemon is None:
        forbidden_pokemon = []
    # turn of gradient calculation
    model.eval()
    t = len(chosen_pokemon)
    candidates = {tuple(chosen_pokemon): 1}
    for i in tqdm(range(t, 6), desc="beam search", leave=True):
        new_candidates = {}
        for candidate in tqdm(candidates.keys(), desc="beam step", leave=False):
            new_candidates.update(beam_step(
                model,
                tokenizer,
                list(candidate),
                format,
                candidates[candidate],
                n_beams=n_beams + 1,
                forbidden_pokemon=forbidden_pokemon
            ))
        candidates = filter_beams(new_candidates, i, n_beams=n_beams)
        if len(candidates) < n_beams:
            warnings.warn(f"Could not find {n_beams} candidates at step {i + 1}, only found {len(candidates)}")

    # return best candidate without a duplicate
    return sorted(candidates.items(), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    base = ['Victini']
    format = 'gen9nationaldex'
    model, tokenizer = load_model()
