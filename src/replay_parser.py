import os

from ban_lists import BAN_LISTS
from data_classes import *


def name_corrections(name):
    corrections = {
        "Tauros-Paldea-Fire": "Tauros-Paldea-Blaze",
        "Tauros-Paldea-Water": "Tauros-Paldea-Aqua",
        "Tauros-Paldea": "Tauros-Paldea-Combat",
        "Tatsugiri": "Tatsugiri",
        "Florges": "Florges",
        "Dudunsparce": "Dudunsparce",
        "Shellos": "Shellos",
        "Gastrodon": "Gastrodon",
        "Unown": "Unown",
        "Vivillon": "Vivillon",
        "Floette": "Floette",
        "Furfrou": "Furfrou",
        "Alcremie": "Alcremie",
        "Deerling": "Deerling",
        "Sawsbuck": "Sawsbuck",

    }
    # remove mega from name
    if "-Mega" in name:
        name = name.split("-Mega")[0]
    for key in corrections:
        if key in name:
            name = corrections[key]
    return name


def parse_replay(format, replay_id):
    filepath = f'replays/{format}/{replay_id}.log'
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    rated = False
    p1_active = ""
    p2_active = ""
    p1 = None
    p2 = None
    winner = None
    contains_banned = False

    try:
        for raw_line in lines:
            # print(f"{replay_id}, {p1_active}, {p2_active}\t{raw_line}")
            split_line = raw_line.split('|')
            if "player" in split_line and len(split_line) > 4:
                if "p1" in split_line and p1 is None:
                    p1_name = split_line[3]
                    try:
                        p1_rating = int(split_line[-1])
                    except ValueError:
                        p1_rating = 1000
                    p1 = Player(p1_name, p1_rating)
                elif "p2" in split_line and p2 is None:
                    p2_name = split_line[3]
                    try:
                        p2_rating = int(split_line[-1])
                    except ValueError:
                        p2_rating = 1000
                    p2 = Player(p2_name, p2_rating)
            if "rated" in split_line:
                rated = True
            if "win" in split_line:
                winner = split_line[2].rstrip()
            if "poke" in split_line:
                if "," in split_line[3]:
                    name = split_line[3].split(', ')[0]
                    gender = split_line[3].split(', ')[1]
                else:
                    name = split_line[3]
                    gender = None
                name = name_corrections(name)
                poke = Pokemon(name, gender)
                if "p1" in split_line:
                    p1.add_pokemon(poke)
                elif "p2" in split_line:
                    p2.add_pokemon(poke)
            if "switch" in split_line:
                name = split_line[3]
                if "," in name:
                    name = name.split(', ')[0]
                if "p1a:" in raw_line:
                    if p1.lead is None:
                        p1.set_lead(name)
                    p1_active = name
                elif "p2a:" in raw_line:
                    if p2.lead is None:
                        p2.set_lead(name)
                    p2_active = name
            if "ability:" in raw_line:
                # find ability name
                for seg in split_line:
                    if "ability:" in seg:
                        ability = seg.split(": ")[1].rstrip()
                        if "p1" in raw_line:
                            p1.set_pokemon_ability(p1_active, ability)
                        elif "p2" in raw_line:
                            p2.set_pokemon_ability(p2_active, ability)
            if "-ability" in raw_line:
                ability = split_line[3].rstrip()
                if "p1" in raw_line:
                    p1.set_pokemon_ability(p1_active, ability)
                elif "p2" in raw_line:
                    p2.set_pokemon_ability(p2_active, ability)
            if "move" in split_line:
                move = split_line[3]
                if "p1" in split_line[2]:
                    p1.add_pokemon_move(p1_active, move)
                elif "p2" in split_line[2]:
                    p2.add_pokemon_move(p2_active, move)
            if "-terastallize" in raw_line:
                if "p1" in raw_line:
                    p1.set_pokemon_tera_type(p1_active, split_line[3].rstrip())
                elif "p2" in raw_line:
                    p2.set_pokemon_tera_type(p2_active, split_line[3].rstrip())
            if "cant" in split_line:
                if "p1" in raw_line and len(split_line) > 4:
                    p1.add_pokemon_move(p1_active, split_line[4].rstrip())
                elif "p2" in raw_line and len(split_line) > 4:
                    p2.add_pokemon_move(p2_active, split_line[4].rstrip())
            if "item:" in raw_line:
                for seg in split_line:
                    if "item:" in seg:
                        item = seg.split(": ")[1].rstrip()
                        if "p1" in raw_line:
                            p1.set_pokemon_item(p1_active, item)
                        elif "p2" in raw_line:
                            p2.set_pokemon_item(p2_active, item)
            if "-item" in raw_line:
                pass
            if "-enditem" in raw_line:
                item = split_line[3]
                if "p1" in raw_line:
                    p1.set_pokemon_item(p1_active, item)
                elif "p2" in raw_line:
                    p2.set_pokemon_item(p2_active, item)
    except Exception as e:
        print(f"Error parsing {replay_id}: {e}")
        print(raw_line)
        return None

    if winner is None:
        print(f"ERROR: {replay_id} has no winner")
        # delete replay file
        os.remove(filepath)
        return None

    if len(p1.team) > 6 or len(p2.team) > 6:
        print(f"ERROR: {replay_id} has more than 6 pokemon per team")
        # delete replay file
        os.remove(filepath)
        return None

    # check against bans
    ban_list = BAN_LISTS[format] if format in BAN_LISTS else []
    pokemon = [p.name for p in p1.team] + [p.name for p in p2.team]
    intersection = set(pokemon).intersection(set(ban_list))
    if len(intersection) > 0:
        contains_banned = True

    rating = int((p1.rating + p2.rating) / 2)
    battle = Battle(
        b_id=replay_id,
        b_format=format,
        rated=rated,
        p1=p1,
        p2=p2,
        winner=winner,
        rating=rating,
        contains_banned=contains_banned,
    )
    return battle


if __name__ == '__main__':
    battle = parse_replay('gen9monotype', 'gen9monotype-1769606826')
    print(battle)
