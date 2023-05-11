import shutil

import numpy as np
from sklearn.metrics import recall_score, f1_score, precision_score
from transformers import TrainingArguments, Trainer, \
    AutoModelForMaskedLM, EarlyStoppingCallback, TrainerCallback, TrainerState, TrainerControl
from transformers.integrations import WandbCallback

from battle_tokenizer import tokenize_dataset
from test_models import *
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
        shuffled_train_dataset = self.tokenizer.shuffle_teams_dataset(self.unmasked_train_dataset)
        np.random.shuffle(shuffled_train_dataset)
        kwargs['train_dataloader'] = DataLoader(mask(shuffled_train_dataset, self.tokenizer, task=self.task),
                                                batch_size=args.per_device_train_batch_size)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # make new dataloader
        DataLoader = torch.utils.data.DataLoader
        shuffled_test_dataset = self.tokenizer.shuffle_teams_dataset(self.unmasked_test_dataset)
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


def team_builder_mask(inputs):
    """
    Masking function for the team builder fine-tuning task
    :param inputs: inputs to mask
    :return: masked inputs
    """
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
    """
    Masking function for the viability fine-tuning task
    :param inputs: inputs to mask
    :return: masked inputs
    """
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
    """
    Base masking function for the pretraining task
    :param inputs: inputs to mask
    :param max_value: maximum value in the vocabulary
    :param formats: formats to mask
    :param task: task to mask for
    :return: masked inputs
    """
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


def no_maks(inputs, **kwargs):
    """
    Sanity check, make sure that masking is indeed working as intended
    """
    inputs["labels"] = inputs["input_ids"][:]
    return inputs


def mask(dataset, tokenizer=None, task='pretrain'):
    """
    Perform masking based on the task
    :param dataset: tokenized dataset
    :param tokenizer: tokenizer to use
    :param task: task to mask for
    :return: masked dataset
    """
    max_value = tokenizer.vocab_size - 1
    formats = tokenizer._encode_list(tokenizer.formats)
    masked = []
    for i, data in enumerate(dataset):
        # restore previous mask
        if 'labels' in data:
            restore_ids(data)
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
    return masked


def train(dataset_train, dataset_test, tokenizer, name=DEFAULT_MODEL, from_saved=False, save_name=None,
          custom_task="pretrain"):
    """
    Train the model
    :param dataset_train: training dataset
    :param dataset_test: testing dataset
    :param tokenizer: tokenizer to use
    :param name: name of the model
    :param from_saved: whether to load from a saved model
    :param save_name: name of the model to load from
    :param custom_task: task to train for
    :return: trained model
    """

    def compute_metrics(eval_pred):
        """
        Compute metrics for the model
        :param eval_pred: evaluation predictions, list of predictions and labels
        :return: metrics
        """
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
    switched = tokenizer.switched_sides_dataset(dataset_train)
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
                      save_name=f'checkpoints/pokemon-team-builder-transformer-deberta10-pretrain/checkpoint-35616')
    else:
        dash_count = task.count('-') + 1
        save_name = "-".join(name.split('-')[:-dash_count]) + "-pretrain"
        save_name = f"checkpoints/{save_name}"
        model = train(dataset_train, dataset_val, tokenizer, name=name, from_saved=True,
                      save_name=save_name, custom_task=task)

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
