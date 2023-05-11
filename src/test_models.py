import copy
from typing import Union

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from battle_tokenizer import tokenize_dataset
from utils import *

import numpy as np


def test_outcome_prediction(dataset, tokenizer, model=None, name: str = DEFAULT_MODEL,
                            test_index: Union[int, str] = 0,
                            batch_size: int = 128):
    if model is None:
        model = Model.from_pretrained(name).to(DEVICE)
    if isinstance(model, str):
        model = Model.from_pretrained(model).to(DEVICE)

    random_pokemon = False
    if isinstance(test_index, str):
        if test_index == "winner":
            test_index = 0
        elif test_index == "format":
            test_index = 1
        elif test_index == "pokemon":
            random_pokemon = True
            team2_start = TEAM_1_START_INDEX + 7
            possible = [i for i in range(TEAM_1_START_INDEX, TEAM_1_START_INDEX + 6)] + \
                       [i for i in range(team2_start, team2_start + 6)]

    test_dataset = []
    true_victory = []
    predicted_victory = []
    for test in dataset:
        test = restore_ids(test)
        if random_pokemon:
            test_index = np.random.choice(possible)
        true_victory.append(test['input_ids'][test_index])
        model_input = copy.deepcopy(test)  # copy the test data
        model_input["input_ids"][test_index] = 4 if test_index >= TEAM_1_START_INDEX else 5
        model_input["input_ids"] = torch.tensor(test["input_ids"]).unsqueeze(0).to(DEVICE)
        if "token_type_ids" in test:
            model_input["token_type_ids"] = torch.tensor(test["token_type_ids"]).unsqueeze(0).to(DEVICE)
        if "position_ids" in test:
            model_input["position_ids"] = torch.tensor(test["position_ids"]).unsqueeze(0).to(DEVICE)
        model_input["attention_mask"] = torch.tensor(test["attention_mask"]).unsqueeze(0).to(DEVICE)
        test_dataset.append(model_input)
    with torch.no_grad():
        for i in range(0, len(test_dataset), batch_size):
            batch = test_dataset[i:i + batch_size]
            input_ids = torch.cat([x["input_ids"] for x in batch], dim=0)
            attention_mask = torch.cat([x["attention_mask"] for x in batch], dim=0)
            if "token_type_ids" in batch[0]:
                token_type_ids = torch.cat([x["token_type_ids"] for x in batch], dim=0)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0]
            logits = logits[:, test_index, :]
            predicted_victory += logits.argmax(dim=1).tolist()

        # score using sklearn
        print(
            classification_report(tokenizer.decode(true_victory), tokenizer.decode(predicted_victory), zero_division=0))


def test_prediction_per_format(dataset, tokenizer, model=None, name: str = 'pokemon-team-builder-transformer',
                               test_index: Union[int, str] = 0):
    if model is None:
        model = Model.from_pretrained(name).to(DEVICE)

    # get all formats from test, from column 3
    formats = [test['input_ids'][1] for test in dataset]
    formats = list(set(formats))
    formats.sort()
    for format in formats:
        # filter on format
        format_dataset = filter(lambda x: x['input_ids'][1] == format, dataset[:])
        format_dataset = list(format_dataset)
        print(f"Testing on {len(format_dataset)} samples from format {tokenizer.decode(format)}")
        test_outcome_prediction(format_dataset, tokenizer, model=model, name=name, test_index=test_index)


def test_viability(dataset, tokenizer, model=None, name: str = 'pokemon-team-builder-transformer'):
    if model is None:
        model = Model.from_pretrained(name).to(DEVICE)

    predicted_viability = []
    true_viability = []
    for test in tqdm(dataset, desc="Testing viability"):
        test = restore_ids(test)
        gt = test['input_ids'][0]
        gt = "win" if gt == 0 else "loss"
        true_viability.append(gt)
        model_input = copy.deepcopy(test)  # copy the test data
        model_input["input_ids"][0] = 4
        team_2_start = model_input["token_type_ids"].index(3)
        for i in range(team_2_start, len(model_input["token_type_ids"])):
            model_input["input_ids"][i] = 4
            model_input["attention_mask"][i] = 1
        model_input["input_ids"] = torch.tensor(test["input_ids"]).unsqueeze(0).to(DEVICE)
        model_input["token_type_ids"] = torch.tensor(test["token_type_ids"]).unsqueeze(0).to(DEVICE)
        model_input["position_ids"] = torch.tensor(test["position_ids"]).unsqueeze(0).to(DEVICE)
        model_input["attention_mask"] = torch.tensor(test["attention_mask"]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(**model_input)
            logits = outputs[0]
            logits = logits[:, 0, :]
            pred = logits.argmax(dim=1).tolist()[0]
            if pred == 0:
                pred = "win"
            elif pred == 1:
                pred = "loss"
            else:
                pred = "other"
            predicted_viability.append(pred)

    # score using sklearn
    print(classification_report(true_viability, predicted_viability, zero_division=0))


def test_team_builder(dataset, tokenizer, model=None, name: str = 'pokemon-team-builder-transformer'):
    if model is None:
        model = Model.from_pretrained(name).to(DEVICE)

    for n_context in range(0, 6):
        print(f"Testing on {n_context} context")
        predicted_mon = []
        true_mon = []
        for test in tqdm(dataset, desc="Testing team builder"):
            team_2_start = test["token_type_ids"].index(3)
            if n_context + 2 >= team_2_start:
                continue
            test = restore_ids(test)
            gt = test['input_ids'][n_context + 2]
            # gts = test['input_ids'][n_context + 2:team_2_start]
            gts = [gt]
            gt = tokenizer.decode(gt)
            gts = [tokenizer.decode(gt) for gt in gts]
            model_input = copy.deepcopy(test)  # copy the test data
            model_input["input_ids"][n_context + 2] = 4
            for i in range(n_context + 3, len(model_input["token_type_ids"])):
                model_input["input_ids"][i] = 2
                model_input["attention_mask"][i] = 0
            model_input["input_ids"] = torch.tensor(test["input_ids"]).unsqueeze(0).to(DEVICE)
            model_input["token_type_ids"] = torch.tensor(test["token_type_ids"]).unsqueeze(0).to(DEVICE)
            model_input["position_ids"] = torch.tensor(test["position_ids"]).unsqueeze(0).to(DEVICE)
            model_input["attention_mask"] = torch.tensor(test["attention_mask"]).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = model(**model_input)
                logits = outputs[0]
                logits = logits[:, n_context + 2, :]
                pred = logits.argmax(dim=1).tolist()[0]
                pred = tokenizer.decode(pred)
                if pred in gts:
                    true_mon.append(pred)
                else:
                    true_mon.append(gt)
                predicted_mon.append(pred)

        # score using sklearn
        print(classification_report(true_mon, predicted_mon, zero_division=0))


if __name__ == "__main__":
    formats = ['gen9ou',
               'gen9monotype',
               'gen9doublesou',
               'gen9vgc2023regulationc',
               'gen9nationaldex',
               'gen9uu',
               'gen9ru',
               'gen9ubers',
               'gen9vgc2023series2',
               'gen9nationaldexmonotype',
               'gen9vgc2023series1',
               'banned']
    task = "pretrain"
    name = f'pokemon-team-builder-transformer-deberta6-{task}'
    seed = 42
    np.random.seed(seed)
    dataset, tokenizer = tokenize_dataset(formats=formats, name=name)

    dataset_train, dataset_test = train_test_split(dataset, test_size=0.05, random_state=seed)
    dataset_train, dataset_val = train_test_split(dataset_train, test_size=0.1, random_state=seed)
    if task == "viability":
        test_viability(dataset_test, name=f"checkpoints/{name}", tokenizer=tokenizer)
    elif task == "team-builder":
        test_team_builder(dataset_test, name=f"checkpoints/{name}", tokenizer=tokenizer)
    elif task == "pretrain":
        test_outcome_prediction(dataset_test, name=f"checkpoints/{name}", test_index=0, tokenizer=tokenizer)
        test_outcome_prediction(dataset_test, name=f"checkpoints/{name}", test_index=1, tokenizer=tokenizer)
        test_outcome_prediction(dataset_test, name=f"checkpoints/{name}", test_index=2, tokenizer=tokenizer)
        #
        test_prediction_per_format(dataset_test, name=f"checkpoints/{name}", test_index=0, tokenizer=tokenizer)
        # test_prediction_per_format(dataset_test, name=f"checkpoints/{name}", test_index=2, tokenizer=tokenizer)