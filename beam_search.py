# finding best team with beam search
import warnings

from utils import *


def beam_step(model, tokenizer, chosen_pokemon, format, prob, n_beams=5):
    candidates = {}
    with torch.no_grad():
        tokens = make_tokens_from_team(chosen_pokemon[:], format)
        ids = make_input_ids(tokens, tokenizer)
        logits = model(**ids).logits
        mask_logits = logits[0, len(chosen_pokemon) + TEAM_1_START_INDEX, :]
        mask_logits = torch.softmax(mask_logits, dim=0)
        top_n = torch.topk(mask_logits, n_beams)
        for i in range(n_beams):
            candidate = chosen_pokemon[:]
            candidate.append(tokenizer.decode(top_n.indices[i]))
            candidates[tuple(candidate)] = top_n.values[i] * prob
    return candidates


def beam_search(model=None, tokenizer=None, chosen_pokemon=None, format='gen9ou', n_beams=5):
    if model is None:
        tokenizer, model = load_model()
    if tokenizer is None:
        tokenizer, _ = load_model()
    if chosen_pokemon is None:
        chosen_pokemon = []
    t = len(chosen_pokemon)
    candidates = {tuple(chosen_pokemon): 1}
    for i in range(t, 6):
        new_candidates = {}
        for candidate in candidates:
            new_candidates.update(beam_step(
                model,
                tokenizer,
                list(candidate),
                format,
                candidates[candidate],
                n_beams=n_beams + 1
            ))
        candidates = new_candidates
        # only keep n_beams candidates
        # remove duplicate teams
        # remove teams with duplicates
        to_keep = {}
        seen = []
        for candidate in sorted(candidates, key=candidates.get, reverse=True):
            sorted_candidate = tuple(sorted(candidate))
            if sorted_candidate not in seen and len(set(candidate)) == i + 1:
                to_keep[candidate] = candidates[candidate]
                seen.append(sorted_candidate)
            if len(to_keep) == n_beams:
                break
        candidates = to_keep
        if len(candidates) < n_beams:
            warnings.warn(f"Could not find {n_beams} candidates at step {i + 1}, only found {len(candidates)}")

    # return best candidate without a duplicate
    return sorted(candidates.items(), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    base = ['Victini']
    format = 'gen9nationaldex'
    print(beam_search(base, format, n_beams=20))
