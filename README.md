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
* Gen 9 National Dex Monotype

### Model
The model that is used is [DeBERTa](https://huggingface.co/docs/transformers/main/en/model_doc/deberta#overview). The tokenizer is my own implementation which can be found in `battle_tokenizer.py`. It simply translates replay files into inputs which are fed to the model.

### Training

The model is trained as a "Masked Language Model". This means that, during training, 15% of the tokens are randomly replaced with the [MASK] token[^1]. The model is then trained to predict the original token. This means that the model can predict the winner of a battle, the format of the battle, and the Pokemon on each team. 

The training dataset is modified to increase the diversity of training samples. Firstly, a sample is added for each replay where the teams are swapped places (player 1 becomes player 2 and vice-versa). 

Each epoch, the input the data is reshuffled and the masking process is re-done. This gives a more diverse dataset for the model to train on.

Training was done with a training set and an unmodified validation set. The validation set was used for early stopping.

[^1]: The masking process is the same as described in the BERT paper. Each token has a 15% chance of being selected for masking. If a token is selected, it is masked with a 80% chance, replaced with a random token with a 10% chance, or left unchanged with a 10% chance. This ensures that the model cannot assume that unmasked tokens are correct.

Number of training samples: 2 * 103650 = 207250

### Results after pre-training
Results were calculated on a held out test set.

#### Predicting the winner

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| P1 Win       | 0.66      | 0.64   | 0.65     | 4036    |
| P2 Win       | 0.64      | 0.66   | 0.65     | 3867    |
|              |           |        |          |         |
| accuracy     |           |        | 0.65     | 7903    |
| macro avg    | 0.65      | 0.65   | 0.65     | 7903    |
| weighted avg | 0.65      | 0.65   | 0.65     | 7903    |

#### Predicting the format

|                         | precision | recall | f1-score | support |
|-------------------------|-----------|--------|----------|---------|
| gen9doublesou           | 0.90      | 0.99   | 0.94     | 549     |
| gen9monotype            | 0.99      | 0.93   | 0.96     | 143     |
| gen9nationaldex         | 0.98      | 0.98   | 0.98     | 1609    |
| gen9nationaldexmonotype | 0.99      | 0.89   | 0.94     | 195     |
| gen9ou                  | 0.99      | 0.98   | 0.98     | 1720    |
| gen9ru                  | 0.98      | 0.99   | 0.99     | 118     |
| gen9ubers               | 1.00      | 0.77   | 0.87     | 299     |
| gen9uu                  | 1.00      | 0.98   | 0.99     | 215     |
| gen9vgc2023series1      | 0.92      | 0.98   | 0.95     | 584     |
| gen9vgc2023series2      | 0.98      | 0.99   | 0.98     | 2471    |
|                         |           |        |          |         |
| accuracy                |           |        | 0.97     | 7903    |
| macro avg               | 0.97      | 0.95   | 0.96     | 7903    |
| weighted avg            | 0.97      | 0.97   | 0.97     | 7903    |

#### Predicting the Pokemon in the first slot

|                        | precision | recall | f1-score | support |
|------------------------|-----------|--------|----------|---------|
| Abomasnow              | 1.00      | 0.99   | 1.00     | 163     |
| Abra                   | 1.00      | 1.00   | 1.00     | 1       |
| Absol                  | 1.00      | 1.00   | 1.00     | 12      |
| Accelgor               | 0.00      | 0.00   | 0.00     | 1       |
| Aegislash              | 1.00      | 0.99   | 0.99     | 67      |
| Aerodactyl             | 1.00      | 0.88   | 0.94     | 25      |
| Aggron                 | 1.00      | 0.96   | 0.98     | 24      |
| Alakazam               | 1.00      | 0.80   | 0.89     | 15      |
| Alcremie               | 0.00      | 0.00   | 0.00     | 1       |
| Alcremie-Lemon-Cream   | 0.00      | 0.00   | 0.00     | 1       |
| Alcremie-Rainbow-Swirl | 0.00      | 0.00   | 0.00     | 1       |
| Alcremie-Ruby-Cream    | 0.00      | 0.00   | 0.00     | 1       |
| Alcremie-Salted-Cream  | 1.00      | 1.00   | 1.00     | 2       |
| Alomomola              | 1.00      | 0.74   | 0.85     | 74      |
| Altaria                | 0.93      | 0.93   | 0.93     | 59      |
| Ambipom                | 1.00      | 1.00   | 1.00     | 6       |
| Amoonguss              | 1.00      | 1.00   | 1.00     | 1039    |
| Ampharos               | 1.00      | 0.77   | 0.87     | 39      |
| Annihilape             | 1.00      | 1.00   | 1.00     | 441     |
| Appletun               | 1.00      | 0.90   | 0.95     | 29      |
| Araquanid              | 1.00      | 1.00   | 1.00     | 25      |
| Arboliva               | 0.99      | 1.00   | 1.00     | 134     |
| Arcanine               | 0.96      | 1.00   | 0.98     | 312     |
| Arcanine-Hisui         | 0.00      | 0.00   | 0.00     | 2       |
| Archeops               | 0.00      | 0.00   | 0.00     | 3       |
| Arctovish              | 1.00      | 0.67   | 0.80     | 3       |
| Arctozolt              | 1.00      | 0.36   | 0.53     | 11      |
| Ariados                | 1.00      | 1.00   | 1.00     | 1       |
| Armaldo                | 1.00      | 0.67   | 0.80     | 6       |
| Armarouge              | 0.94      | 1.00   | 0.97     | 417     |
| Aron                   | 1.00      | 0.40   | 0.57     | 5       |
| Arrokuda               | 0.00      | 0.00   | 0.00     | 1       |
| Articuno               | 0.00      | 0.00   | 0.00     | 2       |
| Articuno-Galar         | 0.00      | 0.00   | 0.00     | 1       |
| Audino                 | 1.00      | 0.50   | 0.67     | 2       |
| Aurorus                | 1.00      | 0.50   | 0.67     | 2       |
| Avalugg                | 1.00      | 0.77   | 0.87     | 35      |
| Avalugg-Hisui          | 0.00      | 0.00   | 0.00     | 1       |
| Azelf                  | 0.00      | 0.00   | 0.00     | 5       |
| Azumarill              | 1.00      | 1.00   | 1.00     | 189     |
| Banette                | 1.00      | 0.95   | 0.98     | 21      |
| Barraskewda            | 1.00      | 0.96   | 0.98     | 68      |
| Basculegion            | 0.00      | 0.00   | 0.00     | 3       |
| Basculin               | 0.00      | 0.00   | 0.00     | 5       |
| Basculin-Blue-Striped  | 1.00      | 1.00   | 1.00     | 1       |
| Bastiodon              | 0.00      | 0.00   | 0.00     | 2       |
| Baxcalibur             | 1.00      | 1.00   | 1.00     | 277     |
| Beartic                | 1.00      | 0.87   | 0.93     | 15      |
| Beedrill               | 1.00      | 1.00   | 1.00     | 26      |
| Beheeyem               | 0.00      | 0.00   | 0.00     | 3       |
| Bellibolt              | 1.00      | 0.88   | 0.94     | 42      |
| Bellossom              | 0.00      | 0.00   | 0.00     | 1       |
| Bewear                 | 1.00      | 0.86   | 0.92     | 7       |
| Bibarel                | 0.00      | 0.00   | 0.00     | 1       |
| Bisharp                | 1.00      | 0.91   | 0.95     | 64      |
| Blacephalon            | 1.00      | 1.00   | 1.00     | 16      |
| Blastoise              | 1.00      | 0.92   | 0.96     | 13      |
| Blaziken               | 1.00      | 1.00   | 1.00     | 37      |
| Blipbug                | 0.00      | 0.00   | 0.00     | 1       |
| Blissey                | 1.00      | 0.97   | 0.98     | 59      |
| Boltund                | 0.00      | 0.00   | 0.00     | 1       |
| Bombirdier             | 1.00      | 0.89   | 0.94     | 27      |
| Bonsly                 | 0.00      | 0.00   | 0.00     | 1       |
| Brambleghast           | 1.00      | 0.91   | 0.95     | 54      |
| Bramblin               | 0.00      | 0.00   | 0.00     | 3       |
| Braviary               | 1.00      | 0.93   | 0.96     | 14      |
| Breloom                | 1.00      | 0.99   | 0.99     | 179     |
| Bronzong               | 1.00      | 0.95   | 0.97     | 40      |
| Brute Bonnet           | 0.99      | 0.99   | 0.99     | 172     |
| Bruxish                | 1.00      | 0.50   | 0.67     | 12      |
| Buizel                 | 0.00      | 0.00   | 0.00     | 1       |
| Bulbasaur              | 0.00      | 0.00   | 0.00     | 1       |
| Butterfree             | 1.00      | 0.75   | 0.86     | 4       |
| Buzzwole               | 1.00      | 0.93   | 0.97     | 15      |
| Cacnea                 | 0.00      | 0.00   | 0.00     | 1       |
| Cacturne               | 1.00      | 0.29   | 0.44     | 7       |
| Camerupt               | 1.00      | 0.88   | 0.93     | 16      |
| Carbink                | 0.00      | 0.00   | 0.00     | 2       |
| Carkol                 | 0.00      | 0.00   | 0.00     | 1       |
| Carracosta             | 0.00      | 0.00   | 0.00     | 1       |
| Castform               | 0.00      | 0.00   | 0.00     | 2       |
| Celebi                 | 0.00      | 0.00   | 0.00     | 5       |
| Celesteela             | 1.00      | 1.00   | 1.00     | 25      |
| Centiskorch            | 0.00      | 0.00   | 0.00     | 2       |
| Ceruledge              | 0.95      | 1.00   | 0.98     | 225     |
| Cetitan                | 1.00      | 0.96   | 0.98     | 27      |
| Cetoddle               | 1.00      | 0.67   | 0.80     | 3       |
| Chandelure             | 1.00      | 1.00   | 1.00     | 6       |
| Chansey                | 1.00      | 0.90   | 0.95     | 31      |
| Charizard              | 0.99      | 1.00   | 1.00     | 137     |
| Charmeleon             | 0.00      | 0.00   | 0.00     | 2       |
| Chesnaught             | 0.00      | 0.00   | 0.00     | 1       |
| Chewtle                | 1.00      | 1.00   | 1.00     | 1       |
| Chi-Yu                 | 0.95      | 1.00   | 0.97     | 55      |
| Chien-Pao              | 1.00      | 0.97   | 0.99     | 34      |
| Cinccino               | 0.00      | 0.00   | 0.00     | 3       |
| Cinderace              | 0.90      | 1.00   | 0.95     | 162     |
| Clauncher              | 0.00      | 0.00   | 0.00     | 1       |
| Clawitzer              | 1.00      | 0.93   | 0.96     | 14      |
| Claydol                | 0.00      | 0.00   | 0.00     | 2       |
| Clefable               | 1.00      | 1.00   | 1.00     | 25      |
| Clefairy               | 0.00      | 0.00   | 0.00     | 1       |
| Clodsire               | 0.99      | 1.00   | 1.00     | 198     |
| Cloyster               | 0.95      | 0.91   | 0.93     | 23      |
| Coalossal              | 1.00      | 0.97   | 0.98     | 29      |
| Cobalion               | 1.00      | 0.33   | 0.50     | 6       |
| Cofagrigus             | 1.00      | 0.83   | 0.91     | 6       |
| Combusken              | 0.00      | 0.00   | 0.00     | 1       |
| Comfey                 | 1.00      | 1.00   | 1.00     | 5       |
| Conkeldurr             | 1.00      | 1.00   | 1.00     | 6       |
| Copperajah             | 1.00      | 0.60   | 0.75     | 10      |
| Corsola                | 0.00      | 0.00   | 0.00     | 2       |
| Corsola-Galar          | 1.00      | 1.00   | 1.00     | 4       |
| Corviknight            | 0.98      | 1.00   | 0.99     | 213     |
| Corvisquire            | 0.00      | 0.00   | 0.00     | 1       |
| Cottonee               | 0.00      | 0.00   | 0.00     | 1       |
| Crabominable           | 1.00      | 1.00   | 1.00     | 4       |
| Crabrawler             | 0.00      | 0.00   | 0.00     | 1       |
| Cradily                | 1.00      | 0.75   | 0.86     | 4       |
| Cramorant              | 0.00      | 0.00   | 0.00     | 1       |
| Crawdaunt              | 1.00      | 1.00   | 1.00     | 4       |
| Cresselia              | 1.00      | 0.89   | 0.94     | 9       |
| Crobat                 | 1.00      | 0.86   | 0.92     | 7       |
| Crocalor               | 1.00      | 0.20   | 0.33     | 5       |
| Crustle                | 1.00      | 1.00   | 1.00     | 2       |
| Cryogonal              | 1.00      | 0.93   | 0.96     | 14      |
| Cubchoo                | 1.00      | 1.00   | 1.00     | 1       |
| Cyclizar               | 0.96      | 1.00   | 0.98     | 75      |
| Dachsbun               | 1.00      | 0.87   | 0.93     | 15      |
| Darmanitan             | 0.67      | 0.67   | 0.67     | 6       |
| Decidueye              | 1.00      | 0.20   | 0.33     | 5       |
| Dedenne                | 1.00      | 0.67   | 0.80     | 12      |
| Delcatty               | 0.00      | 0.00   | 0.00     | 1       |
| Delibird               | 1.00      | 0.33   | 0.50     | 3       |
| Delphox                | 0.00      | 0.00   | 0.00     | 2       |
| Deoxys-Defense         | 1.00      | 0.67   | 0.80     | 3       |
| Deoxys-Speed           | 0.00      | 0.00   | 0.00     | 2       |
| Dhelmise               | 0.00      | 0.00   | 0.00     | 1       |
| Diancie                | 0.95      | 1.00   | 0.97     | 36      |
| Diggersby              | 1.00      | 0.17   | 0.29     | 6       |
| Ditto                  | 1.00      | 1.00   | 1.00     | 30      |
| Dondozo                | 1.00      | 0.99   | 1.00     | 142     |
| Donphan                | 0.92      | 1.00   | 0.96     | 12      |
| Dracozolt              | 1.00      | 1.00   | 1.00     | 3       |
| Dragalge               | 1.00      | 0.94   | 0.97     | 16      |
| Dragapult              | 0.99      | 1.00   | 1.00     | 185     |
| Dragonite              | 0.99      | 1.00   | 1.00     | 199     |
| Drapion                | 1.00      | 1.00   | 1.00     | 1       |
| Drednaw                | 1.00      | 0.85   | 0.92     | 20      |
| Drifblim               | 1.00      | 0.94   | 0.97     | 16      |
| Drifloon               | 1.00      | 1.00   | 1.00     | 1       |
| Druddigon              | 0.00      | 0.00   | 0.00     | 2       |
| Dubwool                | 0.00      | 0.00   | 0.00     | 2       |
| Dudunsparce            | 1.00      | 1.00   | 1.00     | 15      |
| Dugtrio                | 1.00      | 0.33   | 0.50     | 6       |
| Dugtrio-Alola          | 0.00      | 0.00   | 0.00     | 1       |
| Dunsparce              | 1.00      | 0.50   | 0.67     | 2       |
| Durant                 | 1.00      | 0.67   | 0.80     | 3       |
| Dusknoir               | 0.00      | 0.00   | 0.00     | 2       |
| Eelektrik              | 0.00      | 0.00   | 0.00     | 2       |
| Eelektross             | 1.00      | 0.90   | 0.95     | 10      |
| Eevee                  | 0.00      | 0.00   | 0.00     | 2       |
| Eiscue                 | 1.00      | 0.67   | 0.80     | 3       |
| Electivire             | 1.00      | 0.92   | 0.96     | 12      |
| Electrode              | 1.00      | 0.70   | 0.82     | 10      |
| Empoleon               | 1.00      | 0.75   | 0.86     | 4       |
| Enamorus               | 0.00      | 0.00   | 0.00     | 2       |
| Escavalier             | 0.00      | 0.00   | 0.00     | 1       |
| Espathra               | 1.00      | 1.00   | 1.00     | 30      |
| Espeon                 | 1.00      | 1.00   | 1.00     | 43      |
| Excadrill              | 1.00      | 1.00   | 1.00     | 13      |
| Exeggutor              | 1.00      | 1.00   | 1.00     | 1       |
| Falinks                | 0.00      | 0.00   | 0.00     | 4       |
| Farigiraf              | 0.96      | 1.00   | 0.98     | 46      |
| Feraligatr             | 1.00      | 0.75   | 0.86     | 4       |
| Ferrothorn             | 0.96      | 1.00   | 0.98     | 54      |
| Finizen                | 1.00      | 1.00   | 1.00     | 1       |
| Flamigo                | 1.00      | 1.00   | 1.00     | 22      |
| Flapple                | 0.00      | 0.00   | 0.00     | 2       |
| Flareon                | 1.00      | 1.00   | 1.00     | 1       |
| Fletchinder            | 0.00      | 0.00   | 0.00     | 2       |
| Flittle                | 0.00      | 0.00   | 0.00     | 2       |
| Floatzel               | 0.92      | 0.96   | 0.94     | 23      |
| Florges                | 1.00      | 1.00   | 1.00     | 14      |
| Flutter Mane           | 0.99      | 1.00   | 1.00     | 136     |
| Flygon                 | 1.00      | 1.00   | 1.00     | 1       |
| Forretress             | 1.00      | 0.97   | 0.98     | 30      |
| Froakie                | 0.00      | 0.00   | 0.00     | 1       |
| Froslass               | 1.00      | 0.80   | 0.89     | 5       |
| Frosmoth               | 1.00      | 0.90   | 0.95     | 10      |
| Gallade                | 1.00      | 1.00   | 1.00     | 36      |
| Galvantula             | 1.00      | 1.00   | 1.00     | 9       |
| Garchomp               | 0.86      | 1.00   | 0.92     | 120     |
| Gardevoir              | 1.00      | 1.00   | 1.00     | 18      |
| Garganacl              | 1.00      | 1.00   | 1.00     | 97      |
| Gastly                 | 0.00      | 0.00   | 0.00     | 1       |
| Gastrodon              | 0.97      | 1.00   | 0.99     | 39      |
| Gengar                 | 1.00      | 1.00   | 1.00     | 10      |
| Gholdengo              | 0.90      | 1.00   | 0.95     | 135     |
| Gimmighoul             | 0.00      | 0.00   | 0.00     | 1       |
| Girafarig              | 0.00      | 0.00   | 0.00     | 1       |
| Glaceon                | 1.00      | 0.67   | 0.80     | 3       |
| Glalie                 | 1.00      | 1.00   | 1.00     | 2       |
| Gligar                 | 0.00      | 0.00   | 0.00     | 2       |
| Glimmet                | 1.00      | 0.75   | 0.86     | 4       |
| Glimmora               | 0.95      | 1.00   | 0.98     | 21      |
| Gliscor                | 1.00      | 1.00   | 1.00     | 3       |
| Gogoat                 | 1.00      | 1.00   | 1.00     | 1       |
| Golduck                | 1.00      | 1.00   | 1.00     | 1       |
| Golurk                 | 0.00      | 0.00   | 0.00     | 1       |
| Goodra                 | 1.00      | 1.00   | 1.00     | 8       |
| Gothitelle             | 1.00      | 1.00   | 1.00     | 4       |
| Grafaiai               | 1.00      | 0.93   | 0.97     | 15      |
| Graveler               | 0.00      | 0.00   | 0.00     | 1       |
| Great Tusk             | 0.72      | 1.00   | 0.84     | 51      |
| Greninja               | 1.00      | 1.00   | 1.00     | 19      |
| Grimmsnarl             | 0.89      | 1.00   | 0.94     | 25      |
| Growlithe              | 0.00      | 0.00   | 0.00     | 0       |
| Grumpig                | 0.00      | 0.00   | 0.00     | 1       |
| Gurdurr                | 0.00      | 0.00   | 0.00     | 2       |
| Guzzlord               | 0.00      | 0.00   | 0.00     | 1       |
| Gyarados               | 0.77      | 1.00   | 0.87     | 20      |
| Hariyama               | 1.00      | 1.00   | 1.00     | 3       |
| Hatterene              | 1.00      | 1.00   | 1.00     | 16      |
| Haunter                | 0.00      | 0.00   | 0.00     | 1       |
| Hawlucha               | 0.86      | 1.00   | 0.92     | 6       |
| Haxorus                | 0.86      | 1.00   | 0.92     | 6       |
| Heatran                | 0.75      | 1.00   | 0.86     | 9       |
| Heracross              | 1.00      | 0.89   | 0.94     | 9       |
| Hippowdon              | 1.00      | 1.00   | 1.00     | 3       |
| Honchkrow              | 1.00      | 1.00   | 1.00     | 1       |
| Hoopa-Unbound          | 1.00      | 1.00   | 1.00     | 2       |
| Houndoom               | 1.00      | 1.00   | 1.00     | 3       |
| Houndstone             | 1.00      | 1.00   | 1.00     | 3       |
| Hydreigon              | 0.73      | 1.00   | 0.85     | 11      |
| Indeedee-F             | 1.00      | 1.00   | 1.00     | 3       |
| Infernape              | 0.00      | 0.00   | 0.00     | 0       |
| Inteleon               | 0.00      | 0.00   | 0.00     | 1       |
| Iron Bundle            | 0.95      | 1.00   | 0.97     | 19      |
| Iron Hands             | 0.44      | 1.00   | 0.62     | 12      |
| Iron Jugulis           | 1.00      | 1.00   | 1.00     | 4       |
| Iron Leaves            | 1.00      | 1.00   | 1.00     | 3       |
| Iron Moth              | 0.40      | 1.00   | 0.57     | 8       |
| Iron Thorns            | 1.00      | 1.00   | 1.00     | 4       |
| Iron Treads            | 1.00      | 1.00   | 1.00     | 2       |
| Iron Valiant           | 0.85      | 1.00   | 0.92     | 11      |
| Jigglypuff             | 0.00      | 0.00   | 0.00     | 1       |
| Jolteon                | 1.00      | 1.00   | 1.00     | 4       |
| Joltik                 | 1.00      | 1.00   | 1.00     | 1       |
| Kangaskhan             | 0.00      | 0.00   | 0.00     | 1       |
| Kartana                | 1.00      | 1.00   | 1.00     | 3       |
| Keldeo-Resolute        | 0.00      | 0.00   | 0.00     | 1       |
| Kilowattrel            | 1.00      | 1.00   | 1.00     | 2       |
| Kingambit              | 0.71      | 1.00   | 0.83     | 5       |
| Kingdra                | 1.00      | 1.00   | 1.00     | 2       |
| Klawf                  | 1.00      | 1.00   | 1.00     | 2       |
| Klefki                 | 1.00      | 1.00   | 1.00     | 1       |
| Komala                 | 1.00      | 1.00   | 1.00     | 1       |
| Kommo-o                | 0.00      | 0.00   | 0.00     | 0       |
| Krookodile             | 1.00      | 1.00   | 1.00     | 2       |
| Landorus-Therian       | 1.00      | 1.00   | 1.00     | 2       |
| Lanturn                | 0.00      | 0.00   | 0.00     | 1       |
| Lechonk                | 0.00      | 0.00   | 0.00     | 1       |
| Lokix                  | 1.00      | 1.00   | 1.00     | 6       |
| Lopunny                | 0.25      | 1.00   | 0.40     | 1       |
| Lumineon               | 0.00      | 0.00   | 0.00     | 1       |
| Lurantis               | 1.00      | 1.00   | 1.00     | 1       |
| Luxray                 | 1.00      | 1.00   | 1.00     | 2       |
| Lycanroc-Dusk          | 0.00      | 0.00   | 0.00     | 0       |
| Mabosstiff             | 0.00      | 0.00   | 0.00     | 0       |
| Magearna-Original      | 0.00      | 0.00   | 0.00     | 1       |
| Magneton               | 1.00      | 1.00   | 1.00     | 3       |
| Makuhita               | 0.00      | 0.00   | 0.00     | 0       |
| Marowak-Alola          | 1.00      | 1.00   | 1.00     | 1       |
| Masquerain             | 1.00      | 1.00   | 1.00     | 1       |
| Maushold               | 1.00      | 1.00   | 1.00     | 1       |
| Maushold-Four          | 1.00      | 1.00   | 1.00     | 4       |
| Mawile                 | 1.00      | 1.00   | 1.00     | 1       |
| Meowscarada            | 0.33      | 1.00   | 0.50     | 2       |
| Metagross              | 1.00      | 1.00   | 1.00     | 1       |
| Mew                    | 1.00      | 1.00   | 1.00     | 1       |
| Mimikyu                | 1.00      | 1.00   | 1.00     | 2       |
| Miraidon               | 0.50      | 1.00   | 0.67     | 1       |
| Mismagius              | 1.00      | 1.00   | 1.00     | 1       |
| Munchlax               | 0.00      | 0.00   | 0.00     | 0       |
| Murkrow                | 0.09      | 1.00   | 0.17     | 1       |
| Nacli                  | 0.00      | 0.00   | 0.00     | 1       |
| Naclstack              | 1.00      | 0.50   | 0.67     | 2       |
| Ninetales              | 0.67      | 1.00   | 0.80     | 2       |
| Noivern                | 1.00      | 1.00   | 1.00     | 3       |
| Numel                  | 0.00      | 0.00   | 0.00     | 1       |
| Nymble                 | 1.00      | 1.00   | 1.00     | 1       |
| Octillery              | 1.00      | 1.00   | 1.00     | 3       |
| Oricorio-Pom-Pom       | 1.00      | 1.00   | 1.00     | 1       |
| Oricorio-Sensu         | 1.00      | 1.00   | 1.00     | 1       |
| Palafin                | 1.00      | 1.00   | 1.00     | 1       |
| Palossand              | 1.00      | 1.00   | 1.00     | 1       |
| Pawmot                 | 1.00      | 1.00   | 1.00     | 2       |
| Pelipper               | 0.11      | 1.00   | 0.20     | 1       |
| Polteageist            | 0.00      | 0.00   | 0.00     | 1       |
| Quagsire               | 1.00      | 1.00   | 1.00     | 1       |
| Raichu-Alola           | 1.00      | 1.00   | 1.00     | 1       |
| Regieleki              | 1.00      | 1.00   | 1.00     | 1       |
| Roaring Moon           | 0.00      | 0.00   | 0.00     | 0       |
| Rotom-Heat             | 1.00      | 1.00   | 1.00     | 1       |
| Rotom-Wash             | 0.00      | 0.00   | 0.00     | 0       |
| Salamence              | 0.50      | 1.00   | 0.67     | 1       |
| Scizor                 | 0.09      | 1.00   | 0.17     | 1       |
| Scolipede              | 1.00      | 1.00   | 1.00     | 1       |
| Scovillain             | 0.00      | 0.00   | 0.00     | 0       |
| Scream Tail            | 0.00      | 0.00   | 0.00     | 0       |
| Shuckle                | 1.00      | 1.00   | 1.00     | 3       |
| Slither Wing           | 0.00      | 0.00   | 0.00     | 0       |
| Slowking               | 1.00      | 1.00   | 1.00     | 1       |
| Slowpoke               | 1.00      | 1.00   | 1.00     | 1       |
| Snover                 | 0.00      | 0.00   | 0.00     | 1       |
| Spinda                 | 1.00      | 1.00   | 1.00     | 2       |
| Swampert               | 0.00      | 0.00   | 0.00     | 0       |
| Sylveon                | 0.00      | 0.00   | 0.00     | 0       |
| Talonflame             | 0.00      | 0.00   | 0.00     | 0       |
| Tatsugiri              | 0.00      | 0.00   | 0.00     | 0       |
| Tauros-Paldea-Blaze    | 1.00      | 1.00   | 1.00     | 1       |
| Toedscool              | 0.00      | 0.00   | 0.00     | 0       |
| Torkoal                | 0.00      | 0.00   | 0.00     | 1       |
| Toxapex                | 0.00      | 0.00   | 0.00     | 0       |
| Toxtricity             | 0.00      | 0.00   | 0.00     | 0       |
| Tynamo                 | 0.00      | 0.00   | 0.00     | 0       |
| Tyranitar              | 0.00      | 0.00   | 0.00     | 0       |
| Umbreon                | 0.20      | 1.00   | 0.33     | 1       |
| Venusaur               | 0.00      | 0.00   | 0.00     | 0       |
| Vivillon               | 1.00      | 1.00   | 1.00     | 1       |
| Volcarona              | 0.00      | 0.00   | 0.00     | 0       |
| Wattrel                | 0.00      | 0.00   | 0.00     | 0       |
|                        |           |        |          |         |
| accuracy               |           |        | 0.96     | 7903    |
| macro avg              | 0.65      | 0.62   | 0.62     | 7903    |
| weighted avg           | 0.96      | 0.96   | 0.96     | 7903    |