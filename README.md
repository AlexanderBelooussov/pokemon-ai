# WIP
This repository is a work in progress. You can already use it to generate teams. 

# Pokemon Team Builder
A [Transformer](https://huggingface.co/docs/transformers/index) based model for generating Pokemon teams.

___
## Basic Usage
Requires Python 3.8+

### Installation
#### Linux
```bash
git clone <this repo>
cd <this repo>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Windows
```bash
git clone <this repo>
cd <this repo>
python3 -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Generating Teams
This can be done using the interactive CLI by running:
```bash
python interactive_team_builder.py
```
___
## Description

### Dataset
The dataset consists of replay files downloaded from [Pokemon Showdown](https://replay.pokemonshowdown.com/). The replays are then parsed into a list containing the player that won, the format of the battle, and the two teams that were used.
The formats are:
* Gen 9 OU
* Gen 9 Doubles OU
* Gen 9 Monotype
* Gen 9 VGC 2023 Series 1
* Gen 9 VGC 2023 Series 2
* Gen 9 National Dex OU
* Gen 9 Ubers
* Gen 9 UU
* Gen 9 RU

### Model
The model that is used is [DeBERTa](https://huggingface.co/docs/transformers/main/en/model_doc/deberta#overview). The tokenizer is my own implementation which can be found in `battle_tokenizer.py`. It simply translates replay files into inputs which are fed to the model.

### Training

The model is trained as a "Masked Language Model". This means that, during training, 15% of the tokens are randomly replaced with the [MASK] token[^1]. The model is then trained to predict the original token. This means that the model can predict the winner of a battle, the format of the battle, and the Pokemon on each team. 

The training dataset is modified to increase the diversity of training samples. Firstly, a sample is added for each replay where the teams are swapped places (player 1 becomes player 2 and vice-versa). 

Each epoch, the input the data is reshuffled and the masking process is re-done. This gives a more diverse dataset for the model to train on.

Training was done with a training set and an unmodified validation set. The validation set was used for early stopping.

[^1]: The masking process is the same as described in the BERT paper. Each token has a 15% chance of being selected for masking. If a token is selected, it is masked with a 80% chance, replaced with a random token with a 10% chance, or left unchanged with a 10% chance. This ensures that the model cannot assume that unmasked tokens are correct.


### Results
Results were calculated on a held out test set.

#### Predicting the winner[^2]

[^2]: The current model does not score well when trying to predict the winner. Previous models based on DistilBERT achieved a weighted F1-score above 0.60, but some flaw prevents it from being a good recommender.

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| P1 Win       | 0.43      | 0.42   | 0.42     | 2870    |
| P2 Win       | 0.43      | 0.44   | 0.43     | 2873    |
|              |           |        |          |         |
| accuracy     |           |        | 0.43     | 5743    |
| macro avg    | 0.43      | 0.43   | 0.43     | 5743    |
| weighted avg | 0.43      | 0.43   | 0.43     | 5743    |

#### Predicting the format

|                    | precision | recall | f1-score | support |
|--------------------|-----------|--------|----------|---------|
| gen9doublesou      | 1.00      | 1.00   | 1.00     | 462     |
| gen9monotype       | 1.00      | 1.00   | 1.00     | 357     |
| gen9nationaldex    | 1.00      | 1.00   | 1.00     | 690     |
| gen9ou             | 1.00      | 1.00   | 1.00     | 1692    |
| gen9ru             | 1.00      | 1.00   | 1.00     | 348     |
| gen9ubers          | 1.00      | 1.00   | 1.00     | 331     |
| gen9uu             | 1.00      | 1.00   | 1.00     | 385     |
| gen9vgc2023series1 | 1.00      | 0.99   | 1.00     | 1122    |
| gen9vgc2023series2 | 0.98      | 1.00   | 0.99     | 356     |
|                    |           |        |          |         |
| accuracy           |           |        | 1.00     | 5743    |
| macro avg          | 1.00      | 1.00   | 1.00     | 5743    |
| weighted avg       | 1.00      | 1.00   | 1.00     | 5743    |

#### Predicting the Pokemon in the first slot

|                       | precision | recall | f1-score | support |
|-----------------------|-----------|--------|----------|---------|
| Abomasnow             | 0.99      | 0.97   | 0.98     | 169     |
| Abra                  | 0.00      | 0.00   | 0.00     | 2       |
| Absol                 | 0.00      | 0.00   | 0.00     | 4       |
| Accelgor              | 0.00      | 0.00   | 0.00     | 4       |
| Aegislash             | 1.00      | 1.00   | 1.00     | 7       |
| Aerodactyl            | 0.00      | 0.00   | 0.00     | 12      |
| Aggron                | 1.00      | 0.17   | 0.29     | 18      |
| Alakazam              | 0.00      | 0.00   | 0.00     | 8       |
| Alomomola             | 1.00      | 1.00   | 1.00     | 49      |
| Altaria               | 1.00      | 1.00   | 1.00     | 53      |
| Altaria-Mega          | 0.00      | 0.00   | 0.00     | 1       |
| Ambipom               | 0.00      | 0.00   | 0.00     | 5       |
| Amoonguss             | 0.99      | 1.00   | 1.00     | 581     |
| Ampharos              | 1.00      | 0.26   | 0.42     | 19      |
| Annihilape            | 0.98      | 1.00   | 0.99     | 421     |
| Appletun              | 1.00      | 0.91   | 0.95     | 22      |
| Applin                | 0.00      | 0.00   | 0.00     | 1       |
| Araquanid             | 0.00      | 0.00   | 0.00     | 2       |
| Arbok                 | 1.00      | 0.60   | 0.75     | 5       |
| Arboliva              | 1.00      | 0.99   | 1.00     | 123     |
| Arcanine              | 0.96      | 1.00   | 0.98     | 126     |
| Arcanine-Hisui        | 0.00      | 0.00   | 0.00     | 1       |
| Arctibax              | 0.00      | 0.00   | 0.00     | 1       |
| Arctovish             | 0.00      | 0.00   | 0.00     | 1       |
| Arctozolt             | 1.00      | 0.50   | 0.67     | 2       |
| Ariados               | 0.00      | 0.00   | 0.00     | 1       |
| Armarouge             | 0.95      | 1.00   | 0.98     | 299     |
| Aron                  | 0.00      | 0.00   | 0.00     | 4       |
| Articuno-Galar        | 0.00      | 0.00   | 0.00     | 2       |
| Audino                | 0.00      | 0.00   | 0.00     | 1       |
| Avalugg               | 0.98      | 0.98   | 0.98     | 43      |
| Axew                  | 0.00      | 0.00   | 0.00     | 2       |
| Azelf                 | 0.00      | 0.00   | 0.00     | 3       |
| Azumarill             | 0.96      | 1.00   | 0.98     | 172     |
| Banette               | 1.00      | 0.56   | 0.71     | 9       |
| Barraskewda           | 1.00      | 0.98   | 0.99     | 66      |
| Basculin              | 0.00      | 0.00   | 0.00     | 8       |
| Basculin-Blue-Striped | 0.00      | 0.00   | 0.00     | 4       |
| Baxcalibur            | 1.00      | 1.00   | 1.00     | 260     |
| Beartic               | 1.00      | 1.00   | 1.00     | 11      |
| Beedrill              | 1.00      | 1.00   | 1.00     | 13      |
| Bellibolt             | 1.00      | 1.00   | 1.00     | 36      |
| Bewear                | 1.00      | 0.33   | 0.50     | 3       |
| Binacle               | 0.00      | 0.00   | 0.00     | 1       |
| Bisharp               | 0.97      | 1.00   | 0.98     | 56      |
| Blacephalon           | 1.00      | 0.50   | 0.67     | 8       |
| Blastoise             | 0.00      | 0.00   | 0.00     | 1       |
| Blaziken              | 1.00      | 0.38   | 0.55     | 8       |
| Blissey               | 0.97      | 1.00   | 0.98     | 64      |
| Bombirdier            | 0.96      | 1.00   | 0.98     | 24      |
| Bouffalant            | 0.00      | 0.00   | 0.00     | 2       |
| Brambleghast          | 1.00      | 0.98   | 0.99     | 48      |
| Braviary              | 1.00      | 0.83   | 0.91     | 18      |
| Breloom               | 1.00      | 0.99   | 1.00     | 172     |
| Bronzong              | 0.96      | 1.00   | 0.98     | 45      |
| Bronzor               | 0.00      | 0.00   | 0.00     | 2       |
| Brute Bonnet          | 1.00      | 1.00   | 1.00     | 83      |
| Bruxish               | 1.00      | 0.92   | 0.96     | 13      |
| Bulbasaur             | 0.00      | 0.00   | 0.00     | 1       |
| Bunnelby              | 1.00      | 1.00   | 1.00     | 1       |
| Buzzwole              | 0.00      | 0.00   | 0.00     | 3       |
| Cacnea                | 0.00      | 0.00   | 0.00     | 1       |
| Cacturne              | 1.00      | 0.27   | 0.43     | 11      |
| Camerupt              | 0.00      | 0.00   | 0.00     | 9       |
| Carbink               | 1.00      | 0.50   | 0.67     | 4       |
| Carkol                | 0.00      | 0.00   | 0.00     | 2       |
| Carracosta            | 0.00      | 0.00   | 0.00     | 2       |
| Celebi                | 0.00      | 0.00   | 0.00     | 2       |
| Celesteela            | 1.00      | 0.43   | 0.60     | 7       |
| Ceruledge             | 0.99      | 1.00   | 0.99     | 186     |
| Cetitan               | 1.00      | 1.00   | 1.00     | 30      |
| Chandelure            | 1.00      | 1.00   | 1.00     | 2       |
| Chansey               | 0.96      | 1.00   | 0.98     | 22      |
| Charizard             | 1.00      | 1.00   | 1.00     | 62      |
| Chi-Yu                | 0.99      | 1.00   | 1.00     | 102     |
| Chien-Pao             | 0.99      | 1.00   | 1.00     | 270     |
| Cinderace             | 0.99      | 1.00   | 1.00     | 114     |
| Clauncher             | 0.00      | 0.00   | 0.00     | 1       |
| Clawitzer             | 1.00      | 0.93   | 0.97     | 15      |
| Claydol               | 0.00      | 0.00   | 0.00     | 1       |
| Clefable              | 1.00      | 1.00   | 1.00     | 10      |
| Clodsire              | 0.99      | 1.00   | 0.99     | 161     |
| Cloyster              | 1.00      | 1.00   | 1.00     | 39      |
| Coalossal             | 1.00      | 0.90   | 0.95     | 20      |
| Cobalion              | 0.00      | 0.00   | 0.00     | 2       |
| Cofagrigus            | 1.00      | 1.00   | 1.00     | 1       |
| Combee                | 0.00      | 0.00   | 0.00     | 1       |
| Comfey                | 1.00      | 0.50   | 0.67     | 2       |
| Conkeldurr            | 0.00      | 0.00   | 0.00     | 1       |
| Copperajah            | 1.00      | 1.00   | 1.00     | 4       |
| Corsola               | 0.00      | 0.00   | 0.00     | 1       |
| Corsola-Galar         | 0.00      | 0.00   | 0.00     | 3       |
| Corviknight           | 0.95      | 1.00   | 0.97     | 140     |
| Crabominable          | 1.00      | 1.00   | 1.00     | 3       |
| Crabrawler            | 0.00      | 0.00   | 0.00     | 1       |
| Cradily               | 0.00      | 0.00   | 0.00     | 1       |
| Crawdaunt             | 0.50      | 0.33   | 0.40     | 3       |
| Cresselia             | 0.00      | 0.00   | 0.00     | 9       |
| Crobat                | 0.00      | 0.00   | 0.00     | 3       |
| Crocalor              | 0.00      | 0.00   | 0.00     | 2       |
| Crustle               | 1.00      | 0.75   | 0.86     | 4       |
| Cryogonal             | 1.00      | 1.00   | 1.00     | 8       |
| Cutiefly              | 0.00      | 0.00   | 0.00     | 1       |
| Cyclizar              | 0.98      | 1.00   | 0.99     | 50      |
| Dachsbun              | 0.85      | 0.94   | 0.89     | 18      |
| Decidueye             | 0.00      | 0.00   | 0.00     | 1       |
| Decidueye-Hisui       | 0.00      | 0.00   | 0.00     | 1       |
| Dedenne               | 1.00      | 1.00   | 1.00     | 6       |
| Delibird              | 0.00      | 0.00   | 0.00     | 8       |
| Deoxys-Defense        | 0.00      | 0.00   | 0.00     | 1       |
| Dhelmise              | 0.00      | 0.00   | 0.00     | 1       |
| Diancie               | 0.95      | 1.00   | 0.98     | 20      |
| Diggersby             | 0.00      | 0.00   | 0.00     | 1       |
| Ditto                 | 0.96      | 1.00   | 0.98     | 25      |
| Dondozo               | 0.97      | 1.00   | 0.99     | 78      |
| Donphan               | 1.00      | 1.00   | 1.00     | 30      |
| Dracozolt             | 0.00      | 0.00   | 0.00     | 3       |
| Dragalge              | 1.00      | 1.00   | 1.00     | 6       |
| Dragapult             | 0.88      | 1.00   | 0.94     | 103     |
| Dragonair             | 0.00      | 0.00   | 0.00     | 2       |
| Dragonite             | 0.98      | 1.00   | 0.99     | 105     |
| Drakloak              | 0.00      | 0.00   | 0.00     | 1       |
| Drapion               | 1.00      | 0.25   | 0.40     | 4       |
| Drednaw               | 1.00      | 1.00   | 1.00     | 12      |
| Drifblim              | 1.00      | 1.00   | 1.00     | 13      |
| Drifloon              | 0.00      | 0.00   | 0.00     | 1       |
| Dudunsparce-*         | 1.00      | 1.00   | 1.00     | 18      |
| Dugtrio               | 0.00      | 0.00   | 0.00     | 6       |
| Dunsparce             | 1.00      | 1.00   | 1.00     | 1       |
| Eelektrik             | 0.00      | 0.00   | 0.00     | 1       |
| Eelektross            | 1.00      | 1.00   | 1.00     | 8       |
| Electivire            | 1.00      | 0.14   | 0.25     | 7       |
| Electrode             | 0.00      | 0.00   | 0.00     | 2       |
| Enamorus-Therian      | 0.00      | 0.00   | 0.00     | 2       |
| Espathra              | 1.00      | 1.00   | 1.00     | 15      |
| Espeon                | 0.97      | 1.00   | 0.99     | 34      |
| Excadrill             | 1.00      | 0.50   | 0.67     | 6       |
| Exeggutor             | 0.00      | 0.00   | 0.00     | 2       |
| Falinks               | 0.00      | 0.00   | 0.00     | 1       |
| Farigiraf             | 0.89      | 1.00   | 0.94     | 25      |
| Feraligatr            | 0.00      | 0.00   | 0.00     | 1       |
| Ferrothorn            | 0.88      | 1.00   | 0.93     | 21      |
| Fidough               | 0.00      | 0.00   | 0.00     | 2       |
| Flamigo               | 0.97      | 1.00   | 0.98     | 29      |
| Flapple               | 0.00      | 0.00   | 0.00     | 1       |
| Flareon               | 1.00      | 1.00   | 1.00     | 1       |
| Flittle               | 0.00      | 0.00   | 0.00     | 1       |
| Floatzel              | 1.00      | 1.00   | 1.00     | 21      |
| Florges               | 1.00      | 1.00   | 1.00     | 9       |
| Florges-Blue          | 0.00      | 0.00   | 0.00     | 1       |
| Florges-White         | 1.00      | 0.17   | 0.29     | 6       |
| Flutter Mane          | 0.97      | 1.00   | 0.98     | 61      |
| Fomantis              | 0.00      | 0.00   | 0.00     | 2       |
| Forretress            | 1.00      | 1.00   | 1.00     | 30      |
| Froslass              | 1.00      | 1.00   | 1.00     | 7       |
| Frosmoth              | 1.00      | 1.00   | 1.00     | 7       |
| Fuecoco               | 0.00      | 0.00   | 0.00     | 2       |
| Gabite                | 1.00      | 0.50   | 0.67     | 2       |
| Gallade               | 0.91      | 1.00   | 0.96     | 32      |
| Galvantula            | 0.00      | 0.00   | 0.00     | 1       |
| Garchomp              | 0.99      | 1.00   | 0.99     | 87      |
| Gardevoir             | 0.93      | 1.00   | 0.97     | 14      |
| Garganacl             | 0.92      | 1.00   | 0.96     | 35      |
| Gastly                | 0.00      | 0.00   | 0.00     | 1       |
| Gastrodon             | 1.00      | 1.00   | 1.00     | 16      |
| Gastrodon-East        | 1.00      | 1.00   | 1.00     | 5       |
| Gengar                | 1.00      | 1.00   | 1.00     | 15      |
| Gholdengo             | 0.90      | 1.00   | 0.95     | 70      |
| Glaceon               | 1.00      | 1.00   | 1.00     | 4       |
| Glalie                | 1.00      | 0.50   | 0.67     | 2       |
| Glimmet               | 1.00      | 0.50   | 0.67     | 2       |
| Glimmora              | 0.42      | 1.00   | 0.59     | 13      |
| Gliscor               | 1.00      | 1.00   | 1.00     | 2       |
| Gogoat                | 1.00      | 1.00   | 1.00     | 1       |
| Golduck               | 1.00      | 1.00   | 1.00     | 1       |
| Goodra                | 1.00      | 1.00   | 1.00     | 4       |
| Grafaiai              | 1.00      | 1.00   | 1.00     | 11      |
| Great Tusk            | 0.80      | 1.00   | 0.89     | 32      |
| Greninja              | 0.40      | 1.00   | 0.57     | 2       |
| Grimmsnarl            | 0.81      | 1.00   | 0.90     | 22      |
| Gumshoos              | 0.00      | 0.00   | 0.00     | 2       |
| Gyarados              | 1.00      | 1.00   | 1.00     | 7       |
| Hariyama              | 0.83      | 1.00   | 0.91     | 5       |
| Hatterene             | 0.86      | 1.00   | 0.92     | 6       |
| Hawlucha              | 1.00      | 1.00   | 1.00     | 3       |
| Haxorus               | 0.62      | 1.00   | 0.76     | 8       |
| Heatran               | 0.12      | 1.00   | 0.21     | 2       |
| Heracross             | 1.00      | 1.00   | 1.00     | 2       |
| Honchkrow             | 1.00      | 1.00   | 1.00     | 1       |
| Houndoom              | 0.00      | 0.00   | 0.00     | 2       |
| Houndstone            | 0.78      | 1.00   | 0.88     | 7       |
| Hydreigon             | 0.50      | 1.00   | 0.67     | 4       |
| Indeedee              | 1.00      | 1.00   | 1.00     | 1       |
| Indeedee-F            | 0.75      | 1.00   | 0.86     | 3       |
| Iron Bundle           | 0.75      | 1.00   | 0.86     | 6       |
| Iron Hands            | 1.00      | 1.00   | 1.00     | 9       |
| Iron Jugulis          | 1.00      | 1.00   | 1.00     | 3       |
| Iron Moth             | 0.75      | 1.00   | 0.86     | 3       |
| Iron Thorns           | 1.00      | 1.00   | 1.00     | 4       |
| Iron Treads           | 1.00      | 1.00   | 1.00     | 3       |
| Iron Valiant          | 0.50      | 1.00   | 0.67     | 3       |
| Jigglypuff            | 0.00      | 0.00   | 0.00     | 1       |
| Jirachi               | 1.00      | 1.00   | 1.00     | 1       |
| Jolteon               | 1.00      | 1.00   | 1.00     | 4       |
| Kartana               | 1.00      | 1.00   | 1.00     | 4       |
| Kecleon               | 0.00      | 0.00   | 0.00     | 2       |
| Kilowattrel           | 0.67      | 1.00   | 0.80     | 4       |
| Kingambit             | 0.13      | 1.00   | 0.24     | 2       |
| Kingdra               | 1.00      | 0.50   | 0.67     | 2       |
| Klefki                | 1.00      | 1.00   | 1.00     | 2       |
| Komala                | 1.00      | 1.00   | 1.00     | 1       |
| Koraidon              | 0.00      | 0.00   | 0.00     | 0       |
| Krookodile            | 1.00      | 1.00   | 1.00     | 2       |
| Landorus-Therian      | 0.25      | 1.00   | 0.40     | 1       |
| Lilligant             | 0.00      | 0.00   | 0.00     | 0       |
| Lokix                 | 1.00      | 1.00   | 1.00     | 6       |
| Luxray                | 0.00      | 0.00   | 0.00     | 1       |
| Mabosstiff            | 1.00      | 1.00   | 1.00     | 1       |
| Magby                 | 0.00      | 0.00   | 0.00     | 0       |
| Magnemite             | 0.00      | 0.00   | 0.00     | 1       |
| Magneton              | 1.00      | 1.00   | 1.00     | 5       |
| Magnezone             | 1.00      | 1.00   | 1.00     | 2       |
| Marowak-Alola         | 0.00      | 0.00   | 0.00     | 0       |
| Maushold              | 1.00      | 1.00   | 1.00     | 1       |
| Maushold-Four         | 1.00      | 1.00   | 1.00     | 4       |
| Meowscarada           | 0.33      | 1.00   | 0.50     | 4       |
| Mimikyu               | 1.00      | 1.00   | 1.00     | 1       |
| Miraidon              | 0.00      | 0.00   | 0.00     | 0       |
| Morgrem               | 0.00      | 0.00   | 0.00     | 1       |
| Mudsdale              | 1.00      | 1.00   | 1.00     | 1       |
| Muk-Alola             | 0.00      | 0.00   | 0.00     | 1       |
| Murkrow               | 0.50      | 1.00   | 0.67     | 1       |
| Nacli                 | 0.00      | 0.00   | 0.00     | 1       |
| Octillery             | 1.00      | 1.00   | 1.00     | 5       |
| Oricorio-Pom-Pom      | 1.00      | 1.00   | 1.00     | 1       |
| Oricorio-Sensu        | 1.00      | 1.00   | 1.00     | 1       |
| Orthworm              | 0.25      | 1.00   | 0.40     | 2       |
| Palafin               | 1.00      | 1.00   | 1.00     | 1       |
| Palossand             | 0.00      | 0.00   | 0.00     | 0       |
| Pawmi                 | 1.00      | 1.00   | 1.00     | 2       |
| Pelipper              | 0.00      | 0.00   | 0.00     | 0       |
| Primeape              | 0.00      | 0.00   | 0.00     | 0       |
| Quaquaval             | 0.00      | 0.00   | 0.00     | 0       |
| Rabsca                | 0.00      | 0.00   | 0.00     | 0       |
| Rillaboom             | 0.00      | 0.00   | 0.00     | 0       |
| Roaring Moon          | 0.00      | 0.00   | 0.00     | 1       |
| Rookidee              | 1.00      | 1.00   | 1.00     | 1       |
| Rotom-Wash            | 0.00      | 0.00   | 0.00     | 0       |
| Sandy Shocks          | 0.17      | 1.00   | 0.29     | 1       |
| Scizor                | 0.00      | 0.00   | 0.00     | 0       |
| Scovillain            | 0.00      | 0.00   | 0.00     | 0       |
| Skarmory              | 0.00      | 0.00   | 0.00     | 0       |
| Skeledirge            | 0.00      | 0.00   | 0.00     | 0       |
| Spinda                | 0.00      | 0.00   | 0.00     | 2       |
| Spiritomb             | 1.00      | 1.00   | 1.00     | 1       |
| Staryu                | 0.00      | 0.00   | 0.00     | 0       |
| Sudowoodo             | 0.00      | 0.00   | 0.00     | 1       |
| Swampert              | 0.00      | 0.00   | 0.00     | 0       |
| Sylveon               | 0.00      | 0.00   | 0.00     | 0       |
| Talonflame            | 0.00      | 0.00   | 0.00     | 0       |
| Tapu Fini             | 1.00      | 1.00   | 1.00     | 1       |
| Tapu Koko             | 0.00      | 0.00   | 0.00     | 0       |
| Tauros-Paldea-Fire    | 1.00      | 1.00   | 1.00     | 1       |
| Tauros-Paldea-Water   | 0.00      | 0.00   | 0.00     | 0       |
| Tinkaton              | 0.00      | 0.00   | 0.00     | 0       |
| Torkoal               | 0.00      | 0.00   | 0.00     | 0       |
| Tornadus-Therian      | 0.00      | 0.00   | 0.00     | 0       |
| Toxapex               | 0.00      | 0.00   | 0.00     | 0       |
| Toxtricity            | 0.00      | 0.00   | 0.00     | 0       |
| Venusaur              | 0.00      | 0.00   | 0.00     | 0       |
| Volcarona             | 0.00      | 0.00   | 0.00     | 0       |
|                       |           |        |          |         |
| accuracy              |           |        | 0.95     | 5743    |
| macro avg             | 0.55      | 0.54   | 0.53     | 5743    |
| weighted avg          | 0.94      | 0.95   | 0.94     | 5743    |