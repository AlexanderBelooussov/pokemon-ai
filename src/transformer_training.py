import copy
import shutil
from typing import Union, List

from sklearn.metrics import recall_score, f1_score, precision_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import TrainingArguments, Trainer, \
    AutoModelForMaskedLM, EarlyStoppingCallback, TrainerCallback, TrainerState, TrainerControl
from transformers.integrations import WandbCallback

from battle_tokenizer import BattleTokenizer, BERTBattleTokenizer
from main import *
from utils import *


# TODO: split data into "brackets" based on elo -> needs more data

class RemaskCallback(TrainerCallback):
    def __init__(self, unmasked_train_dataset, unmasked_test_dataset, tokenizer, task='pretrain'):
        super().__init__()
        self.tokenizer = tokenizer
        self.unmasked_train_dataset = unmasked_train_dataset
        self.unmasked_test_dataset = unmasked_test_dataset
        self.task = task
        self.masked_test_dataset = mask(unmasked_test_dataset, tokenizer, task=task)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # make new dataloader
        DataLoader = torch.utils.data.DataLoader
        shuffled_train_dataset = shuffle_teams_dataset(self.tokenizer, self.unmasked_train_dataset)
        np.random.shuffle(shuffled_train_dataset)
        kwargs['train_dataloader'] = DataLoader(mask(shuffled_train_dataset, self.tokenizer, task=self.task),
                                                batch_size=args.per_device_train_batch_size)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # make new dataloader
        DataLoader = torch.utils.data.DataLoader
        shuffled_test_dataset = shuffle_teams_dataset(self.tokenizer, self.unmasked_test_dataset)
        # kwargs['eval_dataloader'] = DataLoader(self.masked_test_dataset,
        #                                        batch_size=args.per_device_eval_batch_size)
        kwargs['eval_dataloader'] = DataLoader(mask(shuffled_test_dataset, self.tokenizer, task=self.task),
                                               batch_size=args.per_device_eval_batch_size)


class MyTrainer(Trainer):
    def __init__(self, unmasked_train_dataset, unmasked_eval_dataset, custom_tokenizer, custom_task, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unmasked_train_dataset = unmasked_train_dataset
        self.unmasked_eval_dataset = unmasked_eval_dataset
        self.custom_tokenizer = custom_tokenizer
        self.custom_task = custom_task

    def train(self, *args, **kwargs):
        self.eval_dataset = mask(self.unmasked_eval_dataset, self.custom_tokenizer, task=self.custom_task)
        self.train_dataset = mask(self.unmasked_train_dataset, self.custom_tokenizer, task=self.custom_task)
        super().train(*args, **kwargs)


def find_teams_distilbert(inputs):
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


def shuffle_teams_dataset(tokenizer: BattleTokenizer, dataset):
    new_dataset = []
    for data in dataset:
        inputs = data['input_ids'][:]
        if DISTILBERT_LIKE:
            team1, team2 = find_teams_distilbert(data)
            np.random.shuffle(team1)
            np.random.shuffle(team2)
            new_input = inputs[:TEAM_1_START_INDEX] + \
                        team1 + \
                        [tokenizer.token_map['[SEP]']] + \
                        team2 + \
                        [tokenizer.token_map['[SEP]']]
            while len(new_input) < INPUT_LENGTH:
                new_input.append(tokenizer.special_token_map['[PAD]'])
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
                new_input.append(tokenizer.special_token_map['[PAD]'])
            new_dataset.append({'input_ids': new_input,
                                'attention_mask': data['attention_mask'][:],
                                'token_type_ids': data['token_type_ids'][:],
                                'position_ids': data['position_ids'][:]
                                })
        # assert that padding is the same
        assert inputs.count(tokenizer.token_map['[PAD]']) == new_dataset[-1]['input_ids'].count(
            tokenizer.token_map['[PAD]'])
        assert max(new_dataset[-1]['input_ids']) < tokenizer.vocab_size

    return new_dataset


def switched_sides_dataset(tokenizer, dataset):
    switched_dataset = []
    for data in dataset:
        inputs = data['input_ids'][:]
        inputs[0] = 1 if inputs[0] == 0 else 0
        if DISTILBERT_LIKE:
            team1, team2 = find_teams_distilbert(data)
            new_inputs = inputs[:TEAM_1_START_INDEX] + \
                         team2 + \
                         [tokenizer.token_map['[SEP]']] + \
                         team1 + \
                         [tokenizer.token_map['[SEP]']]
            while len(new_inputs) < INPUT_LENGTH:
                new_inputs.append(tokenizer.special_token_map['[PAD]'])

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
                new_inputs.append(tokenizer.token_map['[PAD]'])

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
        assert inputs.count(tokenizer.token_map['[PAD]']) == switched_dataset[-1]['input_ids'].count(
            tokenizer.token_map['[PAD]'])
        assert data['input_ids'][0] != switched_dataset[-1]['input_ids'][0]

    # for i, data in enumerate(switched_dataset):
    #     print(f"Original {i}: {dataset[i]}")
    #     print(f"Switched {i}: {data}")
    return switched_dataset


def tokenize_dataset(formats: List[str], name: str):
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
            for filename in tqdm(os.listdir(f'replays/{format}'), desc=f"Tokenizing {format} replays", leave=False):
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


def restore_ids(inputs):
    if 'labels' not in inputs:
        return inputs
    for j, label in enumerate(inputs['labels']):
        if label != -100:
            inputs['input_ids'][j] = label
    inputs.pop('labels')
    return inputs


def team_builder_mask(inputs):
    team_2_start = inputs["token_type_ids"].index(3)
    # mask_index = np.random.randint(TEAM_1_START_INDEX, team_2_start)
    masked = []
    for mask_index in range(TEAM_1_START_INDEX, team_2_start):
        masked_inputs = copy.deepcopy(inputs)
        masked_inputs["labels"] = [-100 for _ in inputs["input_ids"]]
        for i, token in enumerate(inputs["input_ids"][:]):
            if token == 2 or token == 3:
                continue
            elif i < mask_index:
                continue
            elif i == mask_index:
                masked_inputs["labels"][i] = token
                masked_inputs["input_ids"][i] = 4
            elif i > mask_index:
                masked_inputs["input_ids"][i] = 2
                masked_inputs["attention_mask"][i] = 0
        # print(masked_inputs)
        assert len(masked_inputs["input_ids"]) == len(inputs["input_ids"])
        assert len(masked_inputs["labels"]) == len(inputs["input_ids"])
        assert len(masked_inputs["attention_mask"]) == len(inputs["input_ids"])
        assert len(masked_inputs["token_type_ids"]) == len(inputs["input_ids"])
        masked.append(masked_inputs)
    # return masked
    return np.random.choice(masked)


def viability_mask(inputs):
    team_2_start = inputs["token_type_ids"].index(3)
    # mask_index = np.random.randint(TEAM_1_START_INDEX, team_2_start)
    masked_inputs = copy.deepcopy(inputs)
    masked_inputs["labels"] = [-100 for _ in inputs["input_ids"]]
    masked_inputs["labels"][0] = inputs["input_ids"][0]
    masked_inputs["input_ids"][0] = 4
    for i in range(team_2_start, len(inputs["input_ids"])):
        masked_inputs["input_ids"][i] = 2
        masked_inputs["attention_mask"][i] = 0
    # print(masked_inputs)
    assert len(masked_inputs["input_ids"]) == len(inputs["input_ids"])
    assert len(masked_inputs["labels"]) == len(inputs["input_ids"])
    assert len(masked_inputs["attention_mask"]) == len(inputs["input_ids"])
    assert len(masked_inputs["token_type_ids"]) == len(inputs["input_ids"])
    return masked_inputs


def prob_mask(inputs, max_value, formats, task='pretrain'):
    masked_inputs = copy.deepcopy(inputs)
    masked_inputs["labels"] = [-100 for _ in inputs["input_ids"]]
    for i, token in enumerate(inputs["input_ids"][:]):
        if token == 2 or token == 3:
            continue
        elif (i == 0 or i == 1) and task == 'team-builder':
            continue
        elif i == 1 and task == 'winner':
            continue
        elif i == 0 and task == 'winner':
            masked_inputs["labels"][i] = token
            masked_inputs["input_ids"][i] = 4
        elif np.random.random() < 0.15 and task == 'pretrain':
            masked_inputs["labels"][i] = token
            prob = np.random.random()
            if prob < 0.8:
                masked_inputs["input_ids"][i] = 4
            elif prob < 0.9:
                if i == 0:
                    masked_inputs["input_ids"][i] = 1 if masked_inputs["input_ids"][i] == 0 else 0
                elif i == 1:
                    other_formats = [x for x in formats if x != token]
                    masked_inputs["input_ids"][i] = np.random.choice(other_formats)
                else:
                    masked_inputs["input_ids"][i] = np.random.randint(5, max_value)
    # print(masked_inputs)
    return masked_inputs


def no_maks(inputs, max_value):
    """
    Sanity check, make sure that masking is indeed working as intended
    """
    inputs["labels"] = inputs["input_ids"][:]
    return inputs


def mask(dataset, tokenizer=None, task='pretrain'):
    max_value = tokenizer.vocab_size - 1
    formats = tokenizer._encode_list(tokenizer.formats)
    masked = []
    for i, data in enumerate(dataset):
        # restore previous mask
        if 'labels' in data:
            restore_ids(data)
        # coinflip_mask(data, max_value)
        if task == "team-builder":
            masked_inputs = team_builder_mask(data)
            if isinstance(masked_inputs, list):
                masked.extend(masked_inputs)
            else:
                masked.append(masked_inputs)
        elif task == "viability":
            masked.append(viability_mask(data))
        else:
            masked.append(prob_mask(data, max_value, formats, task))
        # masked.append(no_maks(data, max_value))
        # if i >= 128 and i <= 160:
        #     print(f"Original {i}: {dataset[i]}")
        #     print(f"Masked {i}: {masked[i]}")

    # print(dataset[0])
    # print(masked[0])
    return masked


def train(dataset_train, dataset_test, tokenizer, name=DEFAULT_MODEL, from_saved=False, save_name=None,
          custom_task="pretrain"):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        # Remove ignored index (special tokens)
        true_predictions = []
        true_labels = []
        for i, (preds, labs) in enumerate(zip(predictions, labels)):
            true_predictions.extend([p for p, l in zip(preds, labs) if l != -100])
            true_labels.extend([l for l in labs if l != -100])
        return {
            "weighted_f1": f1_score(true_predictions, true_labels, average='weighted', zero_division=0),
            "macro_f1": f1_score(true_predictions, true_labels, average='macro', zero_division=0),
            "weighted_precision": precision_score(true_predictions, true_labels, average='weighted', zero_division=0),
            "weighted_recall": recall_score(true_predictions, true_labels, average='weighted', zero_division=0),

        }

    # modify the training dataset to add more samples
    switched = switched_sides_dataset(tokenizer, dataset_train)
    dataset_train.extend(switched)
    np.random.shuffle(dataset_train)

    print(f"\nTraining on {len(dataset_train)} samples\n")

    if from_saved:
        if save_name is None:
            save_name = name
        model = Model.from_pretrained(save_name).to(DEVICE)
        model.resize_token_embeddings(tokenizer.vocab_size)
    else:
        config = Config(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.special_token_map["[PAD]"],
            mask_token_id=tokenizer.special_token_map["[MASK]"],
            sep_token_id=tokenizer.special_token_map["[SEP]"],
            early_stopping=True,
            max_position_embeddings=INPUT_LENGTH,
            dropout=0.1,
            attention_dropout=0.1,
        )
        if not DISTILBERT_LIKE:
            config.type_vocab_size = 4
            config.num_hidden_layers = 12
            config.num_attention_heads = 12
            # config.cls_token_id = tokenizer.special_token_map["[CLS]"]
        else:
            config.n_layers = 3
            config.n_heads = 4

        model = AutoModelForMaskedLM.from_config(
            config=config
        ).to(DEVICE)
        if task != "pretrain":
            for param in model.base_model.parameters():
                param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=f"checkpoints/{name}",
        learning_rate=1e-6,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=40,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_steps=100,
        eval_steps=100,
        load_best_model_at_end=True,
        push_to_hub=False,
        # metric_for_best_model="weighted_f1",
        # greater_is_better=True,
        eval_accumulation_steps=10,
    )

    trainer = MyTrainer(
        model=model,
        custom_tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5),
            RemaskCallback(unmasked_train_dataset=dataset_train, unmasked_test_dataset=dataset_test,
                           tokenizer=tokenizer, task=custom_task),
            WandbCallback()
        ],
        compute_metrics=compute_metrics,
        unmasked_train_dataset=dataset_train,
        unmasked_eval_dataset=dataset_test,
        custom_task=custom_task
    )

    trainer.train()
    trainer.save_model(f"checkpoints/{name}")
    # remove checkpoints
    for dir in os.listdir(f"checkpoints/{name}"):
        if dir.startswith('checkpoint'):
            shutil.rmtree(os.path.join(name, dir))
    return model


def test_outcome_prediction(dataset, model=None, name: str = DEFAULT_MODEL,
                            test_index: Union[int, str] = 0,
                            batch_size: int = 128):
    if model is None:
        model = Model.from_pretrained(name).to(DEVICE)

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


def test_prediction_per_format(dataset, model=None, name: str = 'pokemon-team-builder-transformer',
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
        test_outcome_prediction(format_dataset, model, name, test_index)


def test_viability(dataset, model=None, name: str = 'pokemon-team-builder-transformer'):
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


def test_team_builder(dataset, model=None, name: str = 'pokemon-team-builder-transformer'):
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


if __name__ == '__main__':
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
    # pretrain, team-builder, winner, viability
    task = "pretrain"
    name = f'pokemon-team-builder-transformer-deberta10-{task}'
    seed = 42
    np.random.seed(seed)
    dataset, tokenizer = tokenize_dataset(formats=formats, name=name)

    dataset_train, dataset_test = train_test_split(dataset, test_size=0.05, random_state=seed)
    dataset_train, dataset_val = train_test_split(dataset_train, test_size=0.1, random_state=seed)

    if task == "pretrain":
        model = train(dataset_train, dataset_val, tokenizer, name=name, from_saved=True,
                      save_name=f'checkpoints/pokemon-team-builder-transformer-deberta9-pretrain')
    else:
        dash_count = task.count('-') + 1
        save_name = "-".join(name.split('-')[:-dash_count]) + "-pretrain"
        save_name = f"checkpoints/{save_name}"
        model = train(dataset_train, dataset_val, tokenizer, name=name, from_saved=True,
                      save_name=save_name, custom_task=task)

    if task == "viability":
        test_viability(dataset_test, name=f"checkpoints/{name}")
    elif task == "team-builder":
        test_team_builder(dataset_test, name=f"checkpoints/{name}")
    elif task == "pretrain":
        test_outcome_prediction(dataset_test, name=f"checkpoints/{name}", test_index=0)
        test_outcome_prediction(dataset_test, name=f"checkpoints/{name}", test_index=1)
        test_outcome_prediction(dataset_test, name=f"checkpoints/{name}", test_index=2)
        #
        test_prediction_per_format(dataset_test, name=f"checkpoints/{name}", test_index=0)
        # test_prediction_per_format(dataset_test, name=f"checkpoints/{name}", test_index=2)
