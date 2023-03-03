from typing import List, Optional, Union


class Pokemon:
    def __init__(self, name: str,
                 gender=None,
                 ability="",
                 moves=None,
                 spread="",
                 item="",
                 tera=None):
        self.name = name
        self.ability = ability
        self.moves = [] if moves is None else sorted(moves)
        self.spread = spread
        self.item = item
        self.gender = gender
        self.tera = tera

    def add_move(self, move):
        if move not in self.moves:
            self.moves.append(move)
            self.moves.sort()

    def __str__(self):
        output = f"{self.name}"
        if self.gender is not None:
            output += f" ({self.gender})"
        output += f" @ {self.item}\n"
        output += f"\t\tAbility: {self.ability}\n"
        output += f"\t\tTera Type: {self.tera}\n"
        output += f"\t\tEVs: {self.spread}\n"
        for move in self.moves:
            output += f"- {move}\n"
        return output


class Player:
    def __init__(self, name: str, rating: int):
        self.name = name
        self.team = []
        self.rating = rating
        self.lead = None

    def __str__(self):
        output = f"{self.name} ({self.rating})\n"
        for i, poke in enumerate(self.team):
            output += f"\t{'*' if i == self.lead else ' '} {poke}\n"
        return output

    def add_pokemon(self, pokemon: Pokemon):

        self.team.append(pokemon)
        lead = self.team[self.lead] if self.lead is not None else None
        self.team = sorted(self.team, key=lambda x: x.name)
        self.set_lead(lead)

    def set_lead(self, pokemon: Union[Pokemon, str, None]):
        if pokemon is None:
            return
        if isinstance(pokemon, Pokemon):
            self.lead = self.team.index(pokemon)
        elif isinstance(pokemon, str):
            for i, poke in enumerate(self.team):
                if poke.name == pokemon:
                    self.lead = i
                    break

    def get_pokemon(self, pokemon: Union[int, str]) -> Pokemon:
        if isinstance(pokemon, int):
            return self.team[pokemon]
        elif isinstance(pokemon, str):
            for poke in self.team:
                if poke.name == pokemon:
                    return poke
                elif "*" in poke.name and poke.name.split("*")[0][:-1] in pokemon:
                    return poke
                elif "-" in pokemon and pokemon.split("-")[0] in poke.name:
                    return poke
                elif "Mimikyu" in poke.name and "Mimikyu" in pokemon:
                    return poke
                elif "Eiscue" in poke.name and "Eiscue" in pokemon:
                    return poke
            print(f"Could not find pokemon {pokemon} in {self.name}'s team: {[x.name for x in self.team]}")



    def set_pokemon_ability(self, pokemon: Union[int, str], ability: str):
        self.get_pokemon(pokemon).ability = ability

    def add_pokemon_move(self, pokemon: Union[int, str], move: str):
        self.get_pokemon(pokemon).add_move(move)

    def set_pokemon_spread(self, pokemon: Union[int, str], spread: str):
        self.get_pokemon(pokemon).spread = spread

    def set_pokemon_item(self, pokemon: Union[int, str], item: str):
        self.get_pokemon(pokemon).item = item

    def set_pokemon_tera_type(self, pokemon: Union[int, str], tera: str):
        self.get_pokemon(pokemon).tera = tera



class Battle:
    def __init__(self, b_id: str, b_format: str, rated: bool, p1: Player, p2: Player, winner: str, rating: int):
        self.id = b_id
        self.format = b_format
        self.rated = rated
        self.p1 = p1
        self.p2 = p2
        self.winner = winner
        self.rating = rating
        # self.log = log

    def __repr__(self):
        return self.id

    def __str__(self):
        return f"{self.id}\n" \
               f"{self.format}\n" \
               f"{'rated' if self.rated else 'unranked'}\n" \
               f"{self.p1}\n" \
               f"{self.p2}\n" \
               f"winner: {self.winner}\n" \
               f"rating: {self.rating}"
