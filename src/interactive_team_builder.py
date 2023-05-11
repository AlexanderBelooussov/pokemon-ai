import os
import pickle
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM

from beam_search import beam_search_batched
from utils import read_json, find_usage_file, make_tokens_from_team, make_input_ids, TEAM_1_START_INDEX, DEVICE, \
    make_ids_from_team

# MODEL_PATH_BASE = "Zeniph/pokemon-team-builder-transformer-"
# MODEL_PATH_TYPE = "deberta4-large"
MODEL_PATH_BASE = "checkpoints/pokemon-team-builder-transformer-"
VERSION = "deberta6"
MODEL_PATH_TYPE = f"{VERSION}-team-builder"
WIN_PROB_MODEL = f"{VERSION}-pretrain"


# TODO: use data classes and add printing to them
# TODO: Make suggested sets always legal
# TODO: Update usage data

def load_model_interactive():
    for file in os.listdir(f"pickles"):
        if "tokenizer" in file and VERSION in file:
            with open(f"pickles/{file}", 'rb') as f:
                tokenizer = pickle.load(f)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH_BASE + MODEL_PATH_TYPE).to(DEVICE).eval()
    return model, tokenizer


def get_win_prob(model, tokenizer, team, format):
    model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH_BASE + WIN_PROB_MODEL).to(DEVICE).eval()
    ids = make_ids_from_team(team, format, tokenizer)
    ids['input_ids'][0][0] = tokenizer.special_token_map['[MASK]']
    logits = model(**ids).logits
    logits = logits[0, 0, :2]
    logits = torch.softmax(logits, dim=0)
    win_logit = logits[0].item()
    return win_logit


def get_win_prob_batched(model, tokenizer, teams, format):
    model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH_BASE + WIN_PROB_MODEL).to(DEVICE).eval()
    all_inputs = []
    for team in teams:
        ids = make_ids_from_team(team, format, tokenizer)
        ids['input_ids'][0][0] = tokenizer.special_token_map['[MASK]']
        all_inputs.append(ids)
    logits = model(**{
        "input_ids": torch.cat([x["input_ids"] for x in all_inputs], dim=0),
        "attention_mask": torch.cat([x["attention_mask"] for x in all_inputs], dim=0),
        "token_type_ids": torch.cat([x["token_type_ids"] for x in all_inputs], dim=0),
        "position_ids": torch.cat([x["position_ids"] for x in all_inputs], dim=0),
    }).logits
    logits = logits[:, 0, :2]
    logits = torch.softmax(logits, dim=1)
    win_logit = logits[:, 0].tolist()
    return win_logit


def get_team_probability(model, tokenizer, team, format):
    forbidden = set(tokenizer.token_map.keys()) - set(team)
    beam_search_results = beam_search_batched(model, tokenizer, [], format, 20, list(forbidden), silent=True)
    prob = beam_search_results[0][1]
    return prob
    # intermediate = []
    # probs = []
    # for i, pokemon in enumerate(team):
    #     ids = make_ids_from_team(intermediate, format, tokenizer)
    #     logits = model(**ids).logits
    #     logits = logits[0, TEAM_1_START_INDEX + i, :]
    #     logits = torch.softmax(logits, dim=0)
    #     probs.append(logits[tokenizer.encode(pokemon)].item())
    #     intermediate.append(pokemon)
    # print(probs)
    # return np.prod(probs) * np.power(10, power)


def analyse_viability(chosen_pokemon: List[str], format: str):
    try:
        usage_data = read_json('usage_data/' + find_usage_file(format))
    except FileNotFoundError:
        return None

    print("Here is the viability of chosen team:")
    data = read_json('usage_data/' + find_usage_file(format))
    data = data['data']
    viability = []
    usages = []
    for pokemon in chosen_pokemon:
        try:
            v = data[pokemon]['Viability Ceiling']
        except KeyError:
            continue
        usage = data[pokemon]['usage']
        viability.append(v)
        usages.append(usage)
        print(f"{pokemon.ljust(20)}\t{usage:.4f}\t{v}")

    # output column-wise sum, mean, median, min, max
    print(f"\n{'Total viability'.ljust(16)}:\t\t{sum(usages):.4f}\t{np.sum(viability, axis=0)}")
    print(f"{'Mean viability'.ljust(16)}:\t\t{sum(usages) / len(usages):.4f}\t{np.mean(viability, axis=0, dtype=int)}")
    print(f"{'Median viability'.ljust(16)}:\t\t{np.median(usages):.4f}\t{np.median(viability, axis=0).astype(int)}")
    print(f"{'Min viability'.ljust(16)}:\t\t{np.min(usages, axis=0):.4f}\t{np.min(viability, axis=0)}")
    print(f"{'Max viability'.ljust(16)}:\t\t{np.max(usages, axis=0):.4f}\t{np.max(viability, axis=0)}")


def select_from_usage(keys, values, threshold=0.25, required=1, top_k=10):
    new_values = []
    while len(new_values) < required:
        # sort
        new_keys = [k for _, k in sorted(zip(values, keys), reverse=True)]
        new_values = sorted(values, reverse=True)
        # limit to top k
        new_keys = new_keys[:top_k]
        new_values = new_values[:top_k]
        # normalize
        new_values = [v / sum(new_values) for v in new_values]
        # only keep values above 0.25
        new_keys = [k for k, v in zip(new_keys, new_values) if v > threshold]
        new_values = [v for v in new_values if v > threshold]

        threshold *= 0.5
    return new_keys, new_values


def make_set_suggestion_pokemon(pokemon: str, usage_data=None, format: Optional[str] = None):
    if usage_data is None:
        if format is None:
            raise Exception("Either usage_data or format must be specified")
        usage_data = read_json('usage_data/' + find_usage_file(format))
    try:
        data = usage_data['data'][pokemon]
    except KeyError:
        print(f"{pokemon}")
        return None
    pokemon_data = usage_data['data'][pokemon]
    abilities = pokemon_data['Abilities']
    a_scores = list(abilities.values())
    abilities = list(abilities.keys())
    abilities, a_scores = select_from_usage(abilities, a_scores, top_k=3)

    items = pokemon_data['Items']
    i_scores = list(items.values())
    items = list(items.keys())
    items, i_scores = select_from_usage(items, i_scores, required=3, threshold=0.15, top_k=5)

    moves = pokemon_data['Moves']
    m_scores = list(moves.values())
    moves = list(moves.keys())
    moves, m_scores = select_from_usage(moves, m_scores, threshold=0.05, required=4)

    spreads = pokemon_data['Spreads']
    s_scores = list(spreads.values())
    spreads = list(spreads.keys())
    spreads, s_scores = select_from_usage(spreads, s_scores, top_k=5)

    # tera = pokemon_data['Tera']
    # t_scores = list(tera.values())
    # tera = list(tera.keys())
    # tera, t_scores = select_from_usage(tera, t_scores)

    # output in showdown format
    nature = spreads[0].split(':')[0]
    evs = spreads[0].split(':')[1]
    evs = evs.split('/')
    hp = evs[0]
    atk = evs[1]
    df = evs[2]
    spa = evs[3]
    spd = evs[4]
    spe = evs[5]
    print(f"{pokemon} @ {items[0]}")
    print(f"Ability: {abilities[0]}")
    # print(f"Tera Type: {tera[0]}")
    print(f"EVs:", end=" ")
    first = True
    for ev, stat in zip([hp, atk, df, spa, spd, spe], ['HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe']):
        if ev != '0':
            if not first:
                print(f"/ {ev} {stat}", end=" ")
            else:
                print(f"{ev} {stat}", end=" ")
                first = False
    print(f"\n{nature} Nature")
    for move in moves[:4]:
        print(f"- {move}")

    return {
        'abilities': [(a, ascore) for a, ascore in zip(abilities, a_scores)],
        'items': [(i, iscore) for i, iscore in zip(items, i_scores)],
        'moves': [(m, mscore) for m, mscore in zip(moves, m_scores)],
        'spreads': [(s, sscore) for s, sscore in zip(spreads, s_scores)],
        # 'tera': [(t, tscore) for t, tscore in zip(tera, t_scores)]

    }


def generate_random_team(model, tokenizer, format, team, forbidden_pokemon):
    t = team[:]
    if len(team) == 6:
        return t
    ids = make_ids_from_team(t, format, tokenizer)
    logits = model(**ids).logits
    index = TEAM_1_START_INDEX + len(t)
    logits = logits[0, index, :]
    logits = torch.softmax(logits, dim=0).detach().cpu().numpy()
    done = False
    while not done:
        poke_id = np.random.choice(len(logits), p=logits)
        poke_name = tokenizer.decode(poke_id)
        if poke_name not in forbidden_pokemon and poke_name not in t:
            done = True
            t.append(poke_name)
    return generate_random_team(model, tokenizer, format, t, forbidden_pokemon)


def make_set_suggestion_team(format: str, team: List[str]):
    try:
        usage_data = read_json('usage_data/' + find_usage_file(format))
    except FileNotFoundError:
        print(f"Format {format} not found")
        for pokemon in team:
            print(f"{pokemon}\n")
        return None

    print("Here are some suggestions for finalizing your team:")
    suggestions = []
    for pokemon in team:
        suggestion = make_set_suggestion_pokemon(pokemon, usage_data=usage_data)
        suggestions.append(suggestion)
        print("\n")

    # print alternatives
    for i, pokemon in enumerate(team):
        suggestion = suggestions[i]
        if suggestion is None:
            continue
        abilities, items, moves, spreads = suggestion['abilities'], suggestion['items'], suggestion['moves'], \
            suggestion['spreads']
        print(f"{pokemon} Alternatives:")
        if len(abilities) > 1:
            for i, (ability, a_score) in enumerate(abilities):
                print(f"Ability\t{i + 1}.\t{ability.ljust(20)}\t{a_score:.4f}")
            print()
        if len(items) > 1:
            for i, (item, i_score) in enumerate(items):
                print(f"Item\t{i + 1}.\t{item.ljust(20)}\t{i_score:.4f}")
            print()
        if len(moves) > 4:
            for i, (move, m_score) in enumerate(moves):
                print(f"Move\t{i + 1}.\t{move.ljust(20)}\t{m_score:.4f}")
            print()
        if len(spreads) > 1:
            for i, (spread, s_score) in enumerate(spreads):
                print(f"Spread\t{i + 1}.\t{spread.ljust(20)}\t{s_score:.4f}")
            print()
        # if len(tera) > 1:
        #     for i, tera in enumerate(tera[1:]):
        #         print(f"\t{i + 1}.\t{tera}\t{t_scores[i + 1]}")
        #     print()


def run_model(tokenizer, model, chosen_pokemon, format, n_suggestions=20, forbidden_pokemon=None):
    if forbidden_pokemon is None:
        forbidden_pokemon = []
    tokens = make_tokens_from_team(chosen_pokemon[:], format)
    ids = make_input_ids(tokens, tokenizer)
    # print(ids)
    logits = model(**ids).logits
    mask_logits = logits[0, len(chosen_pokemon) + TEAM_1_START_INDEX, :]
    mask_logits = torch.softmax(mask_logits, dim=0)
    topk = torch.topk(mask_logits, n_suggestions + len(chosen_pokemon) + len(forbidden_pokemon))
    pos = 1
    for i, pokemon in enumerate(topk.indices):
        pokemon = tokenizer.decode(pokemon)
        if pokemon in chosen_pokemon or pokemon in forbidden_pokemon:
            continue
        # print(f"\t{i + 1}.\t{pokemon}\tscore: {topk.values[i] * 100:.2f}")
        print(f"\t{pos}.\t{pokemon.ljust(20)}\tscore: {topk.values[i] * 100 :.2f}")
        pos += 1
        if pos > n_suggestions:
            break

    print(f"\nSuggestions for full teams")
    final = []

    # estimate probabilities of given team
    if len(chosen_pokemon) > 0:
        forbidden = set(tokenizer.token_map.keys()) - set(chosen_pokemon)
        bsr = beam_search_batched(model, tokenizer, [], format, 20, list(forbidden), silent=True,
                                  total_steps=len(chosen_pokemon))
        est_chosen_prob = bsr[0][1]
        chosen_pokemon = list(bsr[0][0])
    else:
        est_chosen_prob = 0

    beam_search_results = beam_search_batched(model, tokenizer, chosen_pokemon, format, 200, forbidden_pokemon)
    beam_search_results = beam_search_results[:200]
    win_scores = get_win_prob_batched(None, tokenizer, [x[0] for x in beam_search_results], format)

    for i, (team, score) in tqdm(enumerate(beam_search_results), desc="Evaluating Teams", leave=False,
                                 total=len(beam_search_results)):
        # winl = get_win_prob(model, tokenizer, team, format)
        winl = win_scores[i]
        if len(chosen_pokemon) > 8:
            real_prob_score = get_team_probability(model, tokenizer, team, format)
        else:
            real_prob_score = score + est_chosen_prob
        combined_score = np.log(winl) + real_prob_score
        # combined_score = real_prob_score
        final.append((team, real_prob_score, winl, combined_score))
    ljust_size = max([max([len(pokemon) for pokemon in team]) for team, _ in beam_search_results]) + 2

    final = sorted(final, key=lambda x: x[3], reverse=True)[:10]
    for i, (team, prob_score, win_score, combined_score) in enumerate(final):
        print(f"\t{i + 1}.\t"
              f"{team[0].ljust(ljust_size)}"
              f"{team[1].ljust(ljust_size)}"
              f"{team[2].ljust(ljust_size)}"
              f"{team[3].ljust(ljust_size)}"
              f"{team[4].ljust(ljust_size)}"
              f"{team[5].ljust(ljust_size)}"
              f"\tProb Score: {prob_score:5.2f}"
              f"\tWin Score: {win_score * 100:5.2f}"
              f"\tCombined Score: {combined_score:5.2f}")
    # print()
    # print(f"\tBased on Win Probability:")
    # for i, ((team, prob_score), win_score) in enumerate(sorted(final, key=lambda x: x[1], reverse=True)[:10]):
    #     print(f"\t{i + 1}.\t"
    #           f"{team[0].ljust(ljust_size)}"
    #           f"{team[1].ljust(ljust_size)}"
    #           f"{team[2].ljust(ljust_size)}"
    #           f"{team[3].ljust(ljust_size)}"
    #           f"{team[4].ljust(ljust_size)}"
    #           f"{team[5].ljust(ljust_size)}"
    #           f"\tProb Score: {prob_score * np.power(10, power):5.2f}"
    #           f"\t\tWin Score: {win_score:5.2f}"
    #           f"\t\tCombined: {prob_score*win_score* np.power(10, power):5.2f}")
    return final[0][0]


if __name__ == '__main__':
    model, tokenizer = load_model_interactive()
    print(f"{'=' * 20} Interactive Team Builder {'=' * 20}")
    print("Welcome to the interactive team builder!")
    print("This program will help you build a team for the current generation of competitive pokemon.")
    print("First, please select the format you would like to build a team for.")
    format = None
    while format is None:
        print("\t1. [Gen9] OU")
        print("\t2. [Gen9] DoublesOU")
        print("\t3. [Gen9] Monotype")
        print("\t4. [Gen9] VGC 2023 Series 1")
        print("\t5. [Gen9] National Dex")
        print("\t6. [Gen9] Ubers")
        print("\t7. [Gen9] UU")
        print("\t8. [Gen9] RU")
        print("\t9. [Gen9] VGC 2023 Series 2")
        print("\t10. [Gen9] National Dex Monotype")
        format_map = {
            '1': 'gen9ou',
            '2': 'gen9doublesou',
            '3': 'gen9monotype',
            '4': 'gen9vgc2023regulationc',
            '5': 'gen9nationaldex',
            '6': 'gen9ubers',
            '7': 'gen9uu',
            '8': 'gen9ru',
            '9': 'gen9vgc2023series2',
            '10': 'gen9nationaldexmonotype'
        }
        x = input("> ")
        format = format_map.get(x, None)

    chosen_pokemon = []
    forbidden_pokemon = []
    suggested = None
    while len(chosen_pokemon) < 6:
        print()
        if len(chosen_pokemon):
            print(f"Your current team: {', '.join(chosen_pokemon)}")
        if len(forbidden_pokemon):
            print(f"Forbidden Pokemon: {', '.join(forbidden_pokemon)}")
        print(f"Great! Now, please select enter the {len(chosen_pokemon) + 1}"
              f"{'st' if len(chosen_pokemon) == 0 else 'nd' if len(chosen_pokemon) == 1 else 'rd' if len(chosen_pokemon) == 2 else 'th'} "
              f"pokemon in your team.")
        print("You can either enter the name of the pokemon, or enter one of the options below.")
        print("\tTo forbid a pokemon from being suggested, enter 'forbid <pokemon>'.")
        print("\tTo remove a pokemon from your team, enter 'remove <pokemon>'.")
        print(
            "\t1. If you would like some AI suggestions for the first pokemon in your team, enter 'suggestions' or 1.")
        print("\t2. If you would like to generate a random team, enter 'random' or 2.")
        print("\t3. If you would like to see the usage stats for the current format, enter 'usage' or 3.")
        print("\t4. If you would like to see the viability ceiling for the current format, enter 'viability' or 4.")
        if suggested is not None:
            print("\t5. If you would like to accept the (top) suggested team, enter 'accept' or 5.")
        x = input("> ")
        if x == "usage" or x == "3":
            print("Usage stats are not yet implemented.")
        elif x == "viability" or x == "4":
            print("Viability ceiling is not yet implemented.")
        elif x == "suggestions" or x == "1":
            print(f"Here are some suggestions for the first pokemon in your team:")
            suggested = run_model(tokenizer, model, chosen_pokemon, format, forbidden_pokemon=forbidden_pokemon)
        elif x == "accept" or x == "5":
            if suggested is not None:
                chosen_pokemon = suggested
                suggested = None
            else:
                print("Ask for suggestions first.")
        elif x == "random" or x == "2":
            rand_team = None
            while rand_team is None:
                rand_team = generate_random_team(model, tokenizer, format, chosen_pokemon, forbidden_pokemon)
                print(f"Here is a random team: {', '.join(rand_team)}")
                print("Would you like to accept this team? (y/n)")
                x = input("> ")
                if x == "y":
                    chosen_pokemon = rand_team
                else:
                    rand_team = None
        elif x.startswith("forbid"):
            pokemon = x.split(" ")[1:]
            pokemon = " ".join(pokemon)
            if pokemon in chosen_pokemon:
                print(f"You have already chosen {pokemon}.")
            elif pokemon in forbidden_pokemon:
                print(f"You have already forbidden {pokemon}.")
            else:
                forbidden_pokemon.append(pokemon)
                print(f"Forbidden {pokemon}.")
        elif x.startswith("remove"):
            pokemon = x.split(" ")[1:]
            pokemon = " ".join(pokemon)
            if pokemon in chosen_pokemon:
                chosen_pokemon.remove(pokemon)
                print(f"Removed {pokemon}.")
            else:
                print(f"{pokemon} is not in your team.")
        elif x in tokenizer.get_vocab():
            chosen_pokemon.append(x)
            suggested = None
        else:
            print("Invalid input.")

    print()
    print(f"Your final team: {', '.join(chosen_pokemon)}")
    print()
    make_set_suggestion_team(format, chosen_pokemon)
    analyse_viability(chosen_pokemon, format)
    win_prob = get_win_prob(model, tokenizer, chosen_pokemon, format)
    team_prob = get_team_probability(model, tokenizer, chosen_pokemon, format)
    print(f"CS: {team_prob * win_prob * np.power(10, 6):.0f}")
    print(f"PS: {team_prob * np.power(10, 6):.0f}")
    print(f"WS: {win_prob * 100:.0f}")
    print("\n\nThanks for using the interactive team builder!")
