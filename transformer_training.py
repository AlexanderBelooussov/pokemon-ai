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


# TODO: split data into "brackets" based on elo
# TODO: add lead as token
# TODO: shuffle each epoch

class RemaskCallback(TrainerCallback):
    def __init__(self, unmasked_train_dataset, unmasked_test_dataset, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.unmasked_train_dataset = unmasked_train_dataset
        self.unmasked_test_dataset = unmasked_test_dataset
        self.masked_test_dataset = mask(unmasked_test_dataset, tokenizer)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # make new dataloader
        DataLoader = torch.utils.data.DataLoader
        shuffled_train_dataset = shuffle_teams_dataset(self.tokenizer, self.unmasked_train_dataset)
        np.random.shuffle(shuffled_train_dataset)
        kwargs['train_dataloader'] = DataLoader(mask(shuffled_train_dataset, self.tokenizer),
                                                batch_size=args.per_device_train_batch_size)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # make new dataloader
        DataLoader = torch.utils.data.DataLoader
        shuffled_test_dataset = shuffle_teams_dataset(self.tokenizer, self.unmasked_test_dataset)
        # kwargs['eval_dataloader'] = DataLoader(self.masked_test_dataset,
        #                                        batch_size=args.per_device_eval_batch_size)
        kwargs['eval_dataloader'] = DataLoader(mask(shuffled_test_dataset, self.tokenizer),
                                               batch_size=args.per_device_eval_batch_size)


class MyTrainer(Trainer):
    def __init__(self, unmasked_train_dataset, unmasked_eval_dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unmasked_train_dataset = unmasked_train_dataset
        self.unmasked_eval_dataset = unmasked_eval_dataset

    def train(self, *args, **kwargs):
        self.eval_dataset = mask(self.unmasked_eval_dataset)
        self.train_dataset = mask(self.unmasked_train_dataset)
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
    tokenizer_file = f"pickles/tokenizer.pkl" if DISTILBERT_LIKE else f"pickles/bert_tokenizer.pkl"
    if not os.path.exists(tokenizer_file):
        tokenizer = BattleTokenizer() if DISTILBERT_LIKE else BERTBattleTokenizer()
    else:
        with open(tokenizer_file, 'rb') as f:
            tokenizer = pickle.load(f)

    if not os.path.exists(f'pickles/dataset-{name}.pkl') or not os.path.exists(tokenizer_file):
        tokenizer.rating_dist = {}
        dataset = []
        for format in tqdm(formats, desc=f"Tokenizing formats", leave=False):
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
    # group distribution into 4 buckets
    # rated = [x for x in tokenizer.rating_dist.items() if x[0] != "unrated"]
    # sorted(rated, key=lambda x: x[0], reverse=True)
    # total = sum([x[1] for x in rated])
    # top10percent = int(total * 0.1)
    # top40percent = int(total * 0.4)
    # bucketed = {"high": 0, "mid": 0, "low": 0}
    # current = "high"
    # s = 0
    # thresholds = []
    # for rating, count in rated:
    #     if s > top10percent and current == "high":
    #         current = "mid"
    #         thresholds.append(rating)
    #     if s > top40percent and current == "mid":
    #         current = "low"
    #         thresholds.append(rating)
    #     bucketed[current] += count
    #     s += count
    # bucketed["unrated"] = tokenizer.rating_dist["unrated"]
    # # plot rating distribution
    # print(f"thresholds: {thresholds}")
    # import matplotlib.pyplot as plt
    # plt.bar(bucketed.keys(), bucketed.values(), color='g')
    # plt.show()
    return dataset, tokenizer


def restore_ids(inputs):
    if 'labels' not in inputs:
        return inputs
    for j, label in enumerate(inputs['labels']):
        if label != -100:
            inputs['input_ids'][j] = label
    inputs.pop('labels')
    return inputs


def prob_mask(inputs, max_value):
    masked_inputs = copy.deepcopy(inputs)
    masked_inputs["labels"] = [-100 for _ in inputs["input_ids"]]
    for i, token in enumerate(inputs["input_ids"][:]):
        if token == 2 or token == 3:
            continue

        elif np.random.random() < 0.15:
            masked_inputs["labels"][i] = token
            prob = np.random.random()
            if prob < 0.8:
                masked_inputs["input_ids"][i] = 4
            elif prob < 0.9:
                if i == 0:
                    masked_inputs["input_ids"][i] = 1 if masked_inputs["input_ids"][i] == 0 else 0
                else:
                    masked_inputs["input_ids"][i] = np.random.randint(5, max_value)
    # print(inputs)
    return masked_inputs


def no_maks(inputs, max_value):
    """
    Sanity check, make sure that masking is indeed working as intended
    """
    inputs["labels"] = inputs["input_ids"][:]
    return inputs


def mask(dataset, tokenizer=None):
    # find highest id in dataset
    if tokenizer is None:
        max_value = 0
        for data in dataset:
            max_value = max(max_value, max(data['input_ids']))
    else:
        max_value = tokenizer.vocab_size - 1
    masked = []
    for i, data in enumerate(dataset):
        # restore previous mask
        if 'labels' in data:
            restore_ids(data)
        # coinflip_mask(data, max_value)
        masked.append(prob_mask(data, max_value))
        # masked.append(no_maks(data, max_value))
        # if i >= 128 and i <= 160:
        #     print(f"Original {i}: {dataset[i]}")
        #     print(f"Masked {i}: {masked[i]}")

    # print(dataset[0])
    # print(masked[0])
    return masked


def train(dataset_train, dataset_test, tokenizer, name=DEFAULT_MODEL, from_saved=False, save_name=None):
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
    # shuffles = []
    # for i in range(4):
    #     shuffled = shuffle_teams_dataset(tokenizer, dataset_train)
    #     shuffles.append(shuffled)
    # for shuffle in shuffles:
    #     dataset_train.extend(shuffle)
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
        else:
            config.n_layers = 6
            config.n_heads = 12

        model = AutoModelForMaskedLM.from_config(
            config=config
        ).to(DEVICE)

    training_args = TrainingArguments(
        output_dir=name,
        learning_rate=1e-5,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_train_epochs=200,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_steps=100,
        eval_steps=100,
        load_best_model_at_end=True,
        push_to_hub=False,
        # metric_for_best_model="weighted_f1",
        # greater_is_better=True,
    )

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=10),
            RemaskCallback(unmasked_train_dataset=dataset_train, unmasked_test_dataset=dataset_test,
                           tokenizer=tokenizer),
            WandbCallback()
        ],
        compute_metrics=compute_metrics,
        unmasked_train_dataset=dataset_train,
        unmasked_eval_dataset=dataset_test,
    )

    trainer.train()
    trainer.save_model(name)
    # remove checkpoints
    for dir in os.listdir(name):
        if dir.startswith('checkpoint'):
            shutil.rmtree(os.path.join(name, dir))
    return model


def test_outcome_prediction(dataset, model=None, name: str = DEFAULT_MODEL,
                            test_index: Union[int, str] = 0):
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

    with torch.no_grad():
        true_victory = []
        predicted_victory = []
        for test in dataset:
            test = restore_ids(test)
            if random_pokemon:
                test_index = np.random.choice(possible)
            true_victory.append(test['input_ids'][test_index])
            model_input = copy.deepcopy(test)  # copy the test data
            model_input["input_ids"][test_index] = 4
            model_input["input_ids"] = torch.tensor(test["input_ids"]).unsqueeze(0).to(DEVICE)
            if "token_type_ids" in test:
                model_input["token_type_ids"] = torch.tensor(test["token_type_ids"]).unsqueeze(0).to(DEVICE)
            if "position_ids" in test:
                model_input["position_ids"] = torch.tensor(test["position_ids"]).unsqueeze(0).to(DEVICE)
            model_input["attention_mask"] = torch.tensor(test["attention_mask"]).unsqueeze(0).to(DEVICE)
            output = model(**model_input)
            logits = output.logits
            prediction = torch.argmax(logits, dim=2)[0]
            predicted_victory.append(int(prediction[test_index]))

        # score using sklearn
        print(
            classification_report(tokenizer.decode(true_victory), tokenizer.decode(predicted_victory), zero_division=0))


def test_prediction_per_format(dataset, model=None, name: str = 'pokemon-team-builder-transformer',
                               test_index: Union[int, str] = 0):
    if model is None:
        model = Model.from_pretrained(name).to(DEVICE)

    # get all formats from test, from column 3
    formats = [test['input_ids'][2] for test in dataset]
    formats = list(set(formats))
    formats.sort()
    for format in formats:
        # filter on format
        format_dataset = filter(lambda x: x['input_ids'][2] == format, dataset[:])
        format_dataset = list(format_dataset)
        print(f"Testing on {len(format_dataset)} samples from format {tokenizer.decode(format)}")
        test_outcome_prediction(format_dataset, model, name, test_index)


if __name__ == '__main__':
    formats = ['gen9ou', 'gen9monotype', 'gen9doublesou', 'gen9vgc2023series1', 'gen9nationaldex', 'gen9uu', 'gen9ru',
               'gen9ubers', 'gen9vgc2023series2']
    name = f'pokemon-team-builder-transformer-deberta4-large'
    seed = 42
    np.random.seed(seed)
    dataset, tokenizer = tokenize_dataset(formats=formats, name=name)

    dataset_train, dataset_test = train_test_split(dataset, test_size=0.2, random_state=seed)
    dataset_train, dataset_val = train_test_split(dataset_train, test_size=0.2, random_state=seed)

    # model = train(dataset_train, dataset_val, tokenizer, name=name, from_saved=True,
    #               save_name="pokemon-team-builder-transformer-v2")
    # model = train(dataset_train, dataset_val, tokenizer, name=name, from_saved=False)
    #
    # # dataset_test = mask(dataset_test)
    test_outcome_prediction(dataset_test, name=name, test_index=0)
    test_outcome_prediction(dataset_test, name=name, test_index=1)
    test_outcome_prediction(dataset_test, name=name, test_index=2)
    #
    # test_prediction_per_format(dataset_test, name=name, test_index=0)
    # test_prediction_per_format(dataset_test, name=name, test_index=4)
