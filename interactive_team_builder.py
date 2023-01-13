from typing import List, Optional

import numpy as np

from battle_tokenizer import *
from beam_search import beam_search
from utils import *


# TODO: use data classes and add printing to them
# TODO: Make suggested sets always legal


def analyse_viability(chosen_pokemon: List[str], format: str):
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
        print(f"Could not find {pokemon} in usage data")
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


def make_set_suggestion_team(format: str, team: List[str]):
    usage_data = read_json('usage_data/' + find_usage_file(format))
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
        abilities, items, moves, spreads = suggestion['abilities'], suggestion['items'], suggestion['moves'], suggestion['spreads']
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


def run_model(tokenizer, model, chosen_pokemon, format, n_suggestions=20):
    tokens = make_tokens_from_team(chosen_pokemon[:], format)
    ids = make_input_ids(tokens, tokenizer)
    logits = model(**ids).logits
    mask_logits = logits[0, len(chosen_pokemon) + TEAM_1_START_INDEX, :]
    mask_logits = torch.softmax(mask_logits, dim=0)
    topk = torch.topk(mask_logits, n_suggestions)
    for i, pokemon in enumerate(topk.indices):
        print(f"\t{i + 1}.\t{tokenizer.decode(pokemon).ljust(20)}\tscore: {topk.values[i] * 100:.2f}")

    print(f"\n Suggestions for full teams:")
    beam_search_results = beam_search(model, tokenizer, chosen_pokemon, format, n_suggestions)
    ljust_size = max([max([len(pokemon) for pokemon in team]) for team, _ in beam_search_results]) + 2
    for i, (team, score) in enumerate(beam_search_results[:10]):
        print(f"\t{i + 1}.\t"
              f"{team[0].ljust(ljust_size)}"
              f"{team[1].ljust(ljust_size)}"
              f"{team[2].ljust(ljust_size)}"
              f"{team[3].ljust(ljust_size)}"
              f"{team[4].ljust(ljust_size)}"
              f"{team[5].ljust(ljust_size)}"
              f"\tscore: {score * 100:.2f}")
    return beam_search_results[0][0]


if __name__ == '__main__':
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
        format_map = {
            '1': 'gen9ou',
            '2': 'gen9doublesou',
            '3': 'gen9monotype',
            '4': 'gen9vgc2023series1',
            '5': 'gen9nationaldex',
            '6': 'gen9ubers',
            '7': 'gen9uu',
        }
        x = input("> ")
        format = format_map.get(x, None)

    tokenizer, model = load_model()
    chosen_pokemon = []
    suggested = None
    # chosen_pokemon = ['Dragapult', 'Roaring Moon', 'Iron Valiant', 'Great Tusk', 'Chien-Pao', 'Rotom-Wash']
    while len(chosen_pokemon) < 6:
        print()
        if len(chosen_pokemon):
            print(f"Your current team: {', '.join(chosen_pokemon)}")
        print(f"Great! Now, please select enter the {len(chosen_pokemon) + 1}"
              f"{'st' if len(chosen_pokemon) == 0 else 'nd' if len(chosen_pokemon) == 1 else 'rd' if len(chosen_pokemon) == 2 else 'th'} "
              f"pokemon in your team.")
        print("\t1. If you would like to see the usage stats for the current format, enter 'usage' or 1.")
        print("\t2. If you would like to see the viability ceiling for the current format, enter 'viability' or 2.")
        print("\t3. you would like some suggestions for the first pokemon in your team, enter 'suggestions' or 3.")
        if suggested is not None:
            print("\t4. If you would like to accept the (top) suggested team, enter 'accept' or 4.")
        x = input("> ")
        if x == "usage" or x == "1":
            print("Usage stats are not yet implemented.")
        elif x == "viability" or x == "2":
            print("Viability ceiling is not yet implemented.")
        elif x == "suggestions" or x == "3":
            print(f"Here are some suggestions for the first pokemon in your team:")
            suggested = run_model(tokenizer, model, chosen_pokemon, format)
        elif x == "accept" or x == "4":
            if suggested is not None:
                chosen_pokemon = suggested
                suggested = None
            else:
                print("Ask for suggestions first.")
        elif x in tokenizer.get_vocab():
            chosen_pokemon.append(x)
            suggested = None
        else:
            print("Invalid input.")

    print()
    print(f"Your final team: {', '.join(chosen_pokemon)}")
    print()
    print("Here are some suggestions for finalizing your team:")
    make_set_suggestion_team(format, chosen_pokemon)
    print("Here is the viability of chosen team:")
    analyse_viability(chosen_pokemon, format)
    print("\n\nThanks for using the interactive team builder!")
