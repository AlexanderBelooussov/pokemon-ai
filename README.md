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
* Gen 9 National Dex OU
* Gen 9 Ubers
* Gen 9 UU
* Gen 9 RU

An example set of tokens is:
```
input = ["P1 Win", "[SEP]", "gen9ou", "[SEP]", 
        "Annihilape", "Cyclizar", "Clodsire", "Corviknight", "Volcarona", "Rabsca", "[SEP]", 
        "Tapu Koko", "Greninja", "Iron Moth", "Iron Valiant", "Scizor", "Annihilape", "[SEP]"]
```

### Model
The model that is used is [DistilBERT](https://huggingface.co/docs/transformers/main/en/model_doc/distilbert#overview). The tokenizer is my own implementation which can be found in `battle_tokenizer.py`. It simply translates inputs like the one above into a list of integers.

### Training

The model is trained as a "Masked Language Model". This means that, during training, 15% of the tokens are randomly replaced with the [MASK] token[^1]. The model is then trained to predict the original token. This means that the model can predict the winner of a battle, the format of the battle, and the Pokemon on each team. 

The training dataset is modified to increase the diversity of training samples. Firstly, a sample is added for each replay where the teams are swapped places (player 1 becomes player 2 and vice-versa). Then, for each sample 4 new samples are added with the teams shuffled. This ensures that all Pokemon can appear in any of the 6 slots. This means that we go from x samples to (4 + 1)2x samples.

Each epoch, the input the masking process is re-done. This gives a more diverse dataset for the model to train on.

Training was done with a training set and an unmodified validation set. The validation set was used for early stopping.

[^1]: The masking process is the same as described in the BERT paper. Each token has a 15% chance of being selected for masking. If a token is selected, it is masked with a 80% chance, replaced with a random token with a 10% chance, or left unchanged with a 10% chance. This ensures that the model cannot assume that unmasked tokens are correct.


### Results
Results were calculated on a held out test set.

#### Predicting the winner

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| P1 Win       | 0.64      | 0.57   | 0.60     | 1932    |
| P2 Win       | 0.61      | 0.67   | 0.64     | 1916    |
|              |           |        |          |         |
| accuracy     |           |        | 0.62     | 3848    |
| macro avg    | 0.62      | 0.62   | 0.62     | 3848    |
| weighted avg | 0.62      | 0.62   | 0.62     | 3848    |

#### Predicting the format

|                    | precision | recall | f1-score | support |
|--------------------|-----------|--------|----------|---------|
| gen9doublesou      | 0.99      | 1.00   | 1.00     | 307     |
| gen9monotype       | 0.99      | 1.00   | 1.00     | 306     |
| gen9nationaldex    | 1.00      | 1.00   | 1.00     | 342     |
| gen9ou             | 1.00      | 1.00   | 1.00     | 1260    |
| gen9ru             | 1.00      | 1.00   | 1.00     | 286     |
| gen9ubers          | 1.00      | 0.99   | 1.00     | 253     |
| gen9uu             | 1.00      | 1.00   | 1.00     | 256     |
| gen9vgc2023series1 | 1.00      | 1.00   | 1.00     | 838     |
| accuracy           |           |        | 1.00     | 3848    |
| macro avg          | 1.00      | 1.00   | 1.00     | 3848    |
| weighted avg       | 1.00      | 1.00   | 1.00     | 3848    |

#### Predicting the Pokemon in the first slot

|                       | precision | recall | f1-score | support |
|-----------------------|-----------|--------|----------|---------|
| Abomasnow             | 0.97      | 0.93   | 0.95     | 106     |
| Absol                 | 1.00      | 0.33   | 0.50     | 3       |
| Aegislash             | 1.00      | 0.70   | 0.82     | 10      |
| Aerodactyl            | 1.00      | 0.58   | 0.74     | 12      |
| Aggron                | 1.00      | 0.20   | 0.33     | 5       |
| Alakazam              | 0.50      | 0.25   | 0.33     | 4       |
| Alcremie-Ruby-Cream   | 1.00      | 1.00   | 1.00     | 1       |
| Alomomola             | 0.84      | 0.81   | 0.82     | 26      |
| Altaria               | 0.97      | 0.78   | 0.86     | 45      |
| Altaria-Mega          | 0.00      | 0.00   | 0.00     | 1       |
| Ambipom               | 0.00      | 0.00   | 0.00     | 1       |
| Amoonguss             | 0.93      | 0.96   | 0.94     | 373     |
| Ampharos              | 0.75      | 0.30   | 0.43     | 10      |
| Annihilape            | 0.92      | 0.99   | 0.96     | 302     |
| Appletun              | 1.00      | 0.58   | 0.74     | 12      |
| Araquanid             | 1.00      | 1.00   | 1.00     | 1       |
| Arbok                 | 1.00      | 1.00   | 1.00     | 3       |
| Arboliva              | 0.96      | 0.87   | 0.91     | 90      |
| Arcanine              | 0.89      | 0.96   | 0.92     | 68      |
| Arcanine-Hisui        | 0.00      | 0.00   | 0.00     | 1       |
| Arctovish             | 0.00      | 0.00   | 0.00     | 1       |
| Armarouge             | 0.94      | 0.93   | 0.93     | 184     |
| Aromatisse            | 0.00      | 0.00   | 0.00     | 1       |
| Aron                  | 1.00      | 1.00   | 1.00     | 1       |
| Audino                | 0.00      | 0.00   | 0.00     | 1       |
| Avalugg               | 0.93      | 0.84   | 0.89     | 32      |
| Azelf                 | 1.00      | 0.33   | 0.50     | 6       |
| Azumarill             | 0.93      | 0.96   | 0.94     | 124     |
| Banette               | 1.00      | 0.75   | 0.86     | 4       |
| Barraskewda           | 0.94      | 0.91   | 0.92     | 53      |
| Basculegion-F         | 0.00      | 0.00   | 0.00     | 1       |
| Basculin              | 1.00      | 0.33   | 0.50     | 3       |
| Basculin-Blue-Striped | 1.00      | 0.25   | 0.40     | 4       |
| Bastiodon             | 0.00      | 0.00   | 0.00     | 1       |
| Baxcalibur            | 0.92      | 0.93   | 0.93     | 183     |
| Beartic               | 0.80      | 0.57   | 0.67     | 7       |
| Beedrill              | 1.00      | 0.40   | 0.57     | 5       |
| Bellibolt             | 0.96      | 0.69   | 0.80     | 32      |
| Bewear                | 1.00      | 1.00   | 1.00     | 1       |
| Bisharp               | 0.97      | 0.73   | 0.83     | 41      |
| Blacephalon           | 1.00      | 0.50   | 0.67     | 2       |
| Blastoise             | 1.00      | 1.00   | 1.00     | 1       |
| Blaziken              | 1.00      | 0.50   | 0.67     | 6       |
| Blissey               | 1.00      | 0.86   | 0.92     | 50      |
| Bombirdier            | 0.91      | 0.84   | 0.87     | 25      |
| Bouffalant            | 1.00      | 1.00   | 1.00     | 1       |
| Brambleghast          | 0.92      | 0.74   | 0.82     | 47      |
| Braviary              | 0.84      | 0.89   | 0.86     | 18      |
| Breloom               | 0.97      | 0.86   | 0.91     | 129     |
| Bronzong              | 1.00      | 0.64   | 0.78     | 39      |
| Bronzor               | 0.00      | 0.00   | 0.00     | 1       |
| Brute Bonnet          | 0.98      | 0.88   | 0.92     | 49      |
| Bruxish               | 1.00      | 0.78   | 0.88     | 9       |
| Bulbasaur             | 0.00      | 0.00   | 0.00     | 1       |
| Buzzwole              | 1.00      | 1.00   | 1.00     | 1       |
| Cacnea                | 0.00      | 0.00   | 0.00     | 1       |
| Cacturne              | 1.00      | 1.00   | 1.00     | 2       |
| Camerupt              | 1.00      | 0.73   | 0.84     | 11      |
| Capsakid              | 1.00      | 1.00   | 1.00     | 1       |
| Carbink               | 1.00      | 1.00   | 1.00     | 3       |
| Carkol                | 1.00      | 0.33   | 0.50     | 3       |
| Celebi                | 1.00      | 0.67   | 0.80     | 3       |
| Celesteela            | 1.00      | 0.75   | 0.86     | 4       |
| Ceruledge             | 0.91      | 0.96   | 0.93     | 110     |
| Cetitan               | 0.92      | 0.92   | 0.92     | 13      |
| Chandelure            | 0.00      | 0.00   | 0.00     | 2       |
| Chansey               | 1.00      | 0.69   | 0.82     | 13      |
| Charizard             | 0.88      | 0.90   | 0.89     | 41      |
| Chi-Yu                | 0.95      | 0.95   | 0.95     | 94      |
| Chien-Pao             | 0.90      | 1.00   | 0.95     | 196     |
| Cinderace             | 0.93      | 0.96   | 0.95     | 72      |
| Clauncher             | 1.00      | 1.00   | 1.00     | 1       |
| Clawitzer             | 0.71      | 0.83   | 0.77     | 6       |
| Clefable              | 0.75      | 0.60   | 0.67     | 5       |
| Clodsire              | 0.93      | 0.98   | 0.96     | 100     |
| Cloyster              | 0.90      | 0.82   | 0.86     | 22      |
| Coalossal             | 0.94      | 0.94   | 0.94     | 16      |
| Cofagrigus            | 1.00      | 0.50   | 0.67     | 2       |
| Comfey                | 0.00      | 0.00   | 0.00     | 1       |
| Conkeldurr            | 0.00      | 0.00   | 0.00     | 2       |
| Copperajah            | 1.00      | 0.75   | 0.86     | 4       |
| Corsola-Galar         | 0.00      | 0.00   | 0.00     | 0       |
| Corviknight           | 0.92      | 0.95   | 0.93     | 95      |
| Crabominable          | 0.00      | 0.00   | 0.00     | 2       |
| Crawdaunt             | 1.00      | 1.00   | 1.00     | 1       |
| Cresselia             | 1.00      | 0.80   | 0.89     | 5       |
| Crobat                | 1.00      | 1.00   | 1.00     | 1       |
| Crocalor              | 1.00      | 0.25   | 0.40     | 4       |
| Cryogonal             | 1.00      | 0.50   | 0.67     | 6       |
| Cyclizar              | 0.82      | 0.86   | 0.84     | 37      |
| Dachsbun              | 1.00      | 0.75   | 0.86     | 12      |
| Dedenne               | 1.00      | 0.67   | 0.80     | 3       |
| Delibird              | 1.00      | 0.50   | 0.67     | 2       |
| Diancie               | 1.00      | 1.00   | 1.00     | 3       |
| Diggersby             | 0.00      | 0.00   | 0.00     | 1       |
| Ditto                 | 0.70      | 0.70   | 0.70     | 10      |
| Dolliv                | 0.00      | 0.00   | 0.00     | 0       |
| Dondozo               | 0.89      | 0.92   | 0.90     | 62      |
| Donphan               | 1.00      | 0.89   | 0.94     | 18      |
| Dracozolt             | 1.00      | 1.00   | 1.00     | 2       |
| Dragalge              | 0.71      | 1.00   | 0.83     | 5       |
| Dragapult             | 0.87      | 0.82   | 0.84     | 71      |
| Dragonite             | 0.88      | 0.99   | 0.93     | 82      |
| Drapion               | 1.00      | 1.00   | 1.00     | 1       |
| Drednaw               | 0.90      | 0.75   | 0.82     | 12      |
| Drifblim              | 1.00      | 0.38   | 0.55     | 8       |
| Dudunsparce-*         | 1.00      | 0.83   | 0.91     | 12      |
| Dugtrio               | 1.00      | 0.67   | 0.80     | 3       |
| Eelektrik             | 0.00      | 0.00   | 0.00     | 1       |
| Eelektross            | 1.00      | 0.43   | 0.60     | 7       |
| Eevee                 | 1.00      | 1.00   | 1.00     | 1       |
| Eiscue                | 1.00      | 0.50   | 0.67     | 2       |
| Electivire            | 1.00      | 1.00   | 1.00     | 2       |
| Electrode             | 1.00      | 1.00   | 1.00     | 1       |
| Electrode-Hisui       | 1.00      | 1.00   | 1.00     | 1       |
| Empoleon              | 0.00      | 0.00   | 0.00     | 0       |
| Enamorus-Therian      | 1.00      | 1.00   | 1.00     | 1       |
| Espathra              | 0.86      | 0.60   | 0.71     | 10      |
| Espeon                | 0.81      | 0.95   | 0.88     | 22      |
| Falinks               | 1.00      | 0.33   | 0.50     | 3       |
| Farigiraf             | 0.93      | 0.78   | 0.85     | 18      |
| Fearow                | 0.00      | 0.00   | 0.00     | 1       |
| Ferrothorn            | 0.91      | 1.00   | 0.95     | 10      |
| Fidough               | 0.00      | 0.00   | 0.00     | 1       |
| Flamigo               | 1.00      | 1.00   | 1.00     | 11      |
| Flareon               | 0.00      | 0.00   | 0.00     | 2       |
| Floatzel              | 0.87      | 1.00   | 0.93     | 20      |
| Florges               | 0.67      | 0.33   | 0.44     | 6       |
| Florges-Blue          | 0.00      | 0.00   | 0.00     | 2       |
| Florges-White         | 0.00      | 0.00   | 0.00     | 1       |
| Flutter Mane          | 0.88      | 0.97   | 0.92     | 37      |
| Flygon                | 0.00      | 0.00   | 0.00     | 1       |
| Forretress            | 0.87      | 1.00   | 0.93     | 20      |
| Froslass              | 1.00      | 0.80   | 0.89     | 5       |
| Frosmoth              | 1.00      | 0.44   | 0.62     | 9       |
| Fuecoco               | 1.00      | 1.00   | 1.00     | 1       |
| Gabite                | 0.00      | 0.00   | 0.00     | 1       |
| Gallade               | 0.68      | 0.65   | 0.67     | 20      |
| Galvantula            | 0.00      | 0.00   | 0.00     | 1       |
| Garchomp              | 0.85      | 0.94   | 0.89     | 54      |
| Gardevoir             | 0.86      | 0.86   | 0.86     | 7       |
| Garganacl             | 0.67      | 0.84   | 0.74     | 19      |
| Gastrodon             | 0.71      | 1.00   | 0.83     | 5       |
| Gastrodon-East        | 1.00      | 0.50   | 0.67     | 6       |
| Gengar                | 0.88      | 1.00   | 0.94     | 15      |
| Gholdengo             | 0.68      | 0.90   | 0.78     | 42      |
| Glaceon               | 0.00      | 0.00   | 0.00     | 1       |
| Glalie                | 1.00      | 1.00   | 1.00     | 1       |
| Glimmora              | 1.00      | 0.67   | 0.80     | 9       |
| Gliscor               | 0.00      | 0.00   | 0.00     | 0       |
| Gogoat                | 1.00      | 1.00   | 1.00     | 1       |
| Goodra                | 1.00      | 0.50   | 0.67     | 2       |
| Gothitelle            | 0.00      | 0.00   | 0.00     | 1       |
| Grafaiai              | 0.67      | 0.75   | 0.71     | 8       |
| Great Tusk            | 0.67      | 0.86   | 0.75     | 14      |
| Greedent              | 1.00      | 1.00   | 1.00     | 1       |
| Greninja              | 0.00      | 0.00   | 0.00     | 0       |
| Grimmsnarl            | 0.70      | 0.78   | 0.74     | 9       |
| Gyarados              | 1.00      | 0.67   | 0.80     | 6       |
| Hariyama              | 1.00      | 0.88   | 0.93     | 8       |
| Hatterene             | 0.62      | 1.00   | 0.77     | 5       |
| Hawlucha              | 0.50      | 1.00   | 0.67     | 2       |
| Haxorus               | 0.60      | 0.75   | 0.67     | 4       |
| Heatran               | 0.50      | 1.00   | 0.67     | 1       |
| Heliolisk             | 0.00      | 0.00   | 0.00     | 0       |
| Heracross             | 0.57      | 0.80   | 0.67     | 5       |
| Hippowdon             | 0.00      | 0.00   | 0.00     | 1       |
| Honchkrow             | 1.00      | 0.50   | 0.67     | 2       |
| Houndoom              | 0.00      | 0.00   | 0.00     | 1       |
| Houndstone            | 0.75      | 1.00   | 0.86     | 3       |
| Hydreigon             | 0.50      | 0.67   | 0.57     | 3       |
| Indeedee-F            | 0.33      | 0.33   | 0.33     | 3       |
| Iron Bundle           | 0.83      | 1.00   | 0.91     | 5       |
| Iron Hands            | 0.80      | 1.00   | 0.89     | 8       |
| Iron Jugulis          | 1.00      | 1.00   | 1.00     | 4       |
| Iron Moth             | 0.67      | 1.00   | 0.80     | 2       |
| Iron Thorns           | 0.67      | 1.00   | 0.80     | 2       |
| Iron Treads           | 0.50      | 1.00   | 0.67     | 2       |
| Iron Valiant          | 0.00      | 0.00   | 0.00     | 0       |
| Jolteon               | 0.67      | 1.00   | 0.80     | 2       |
| Kartana               | 1.00      | 1.00   | 1.00     | 2       |
| Kilowattrel           | 0.50      | 0.50   | 0.50     | 2       |
| Kingambit             | 0.00      | 0.00   | 0.00     | 0       |
| Kingdra               | 1.00      | 1.00   | 1.00     | 1       |
| Klefki                | 0.00      | 0.00   | 0.00     | 1       |
| Komala                | 1.00      | 1.00   | 1.00     | 1       |
| Koraidon              | 0.00      | 0.00   | 0.00     | 0       |
| Krookodile            | 0.29      | 1.00   | 0.44     | 2       |
| Larvesta              | 0.00      | 0.00   | 0.00     | 0       |
| Leafeon               | 1.00      | 1.00   | 1.00     | 1       |
| Lokix                 | 0.40      | 1.00   | 0.57     | 2       |
| Lopunny               | 0.00      | 0.00   | 0.00     | 1       |
| Lucario               | 0.00      | 0.00   | 0.00     | 1       |
| Lycanroc              | 0.00      | 0.00   | 0.00     | 0       |
| Lycanroc-Dusk         | 0.00      | 0.00   | 0.00     | 0       |
| Magearna              | 0.00      | 0.00   | 0.00     | 0       |
| Magneton              | 1.00      | 1.00   | 1.00     | 9       |
| Marowak-Alola         | 0.00      | 0.00   | 0.00     | 1       |
| Maushold-Four         | 0.50      | 1.00   | 0.67     | 1       |
| Meowscarada           | 0.83      | 1.00   | 0.91     | 5       |
| Mimikyu               | 0.00      | 0.00   | 0.00     | 0       |
| Miraidon              | 0.00      | 0.00   | 0.00     | 0       |
| Mismagius             | 0.00      | 0.00   | 0.00     | 1       |
| Mudsdale              | 0.00      | 0.00   | 0.00     | 0       |
| Muk-Alola             | 1.00      | 1.00   | 1.00     | 1       |
| Murkrow               | 0.00      | 0.00   | 0.00     | 1       |
| Nacli                 | 0.00      | 0.00   | 0.00     | 1       |
| Noivern               | 0.00      | 0.00   | 0.00     | 0       |
| Octillery             | 1.00      | 1.00   | 1.00     | 1       |
| Oranguru              | 0.00      | 0.00   | 0.00     | 0       |
| Oricorio-Pom-Pom      | 0.00      | 0.00   | 0.00     | 1       |
| Oricorio-Sensu        | 1.00      | 1.00   | 1.00     | 1       |
| Orthworm              | 0.25      | 1.00   | 0.40     | 1       |
| Palafin               | 0.67      | 1.00   | 0.80     | 2       |
| Palossand             | 1.00      | 1.00   | 1.00     | 1       |
| Pelipper              | 0.00      | 0.00   | 0.00     | 0       |
| Pincurchin            | 0.00      | 0.00   | 0.00     | 0       |
| Quaquaval             | 0.00      | 0.00   | 0.00     | 0       |
| Sableye               | 0.00      | 0.00   | 0.00     | 0       |
| Salamence             | 0.00      | 0.00   | 0.00     | 0       |
| Sceptile              | 0.00      | 0.00   | 0.00     | 0       |
| Scizor                | 0.00      | 0.00   | 0.00     | 0       |
| Serperior             | 0.00      | 0.00   | 0.00     | 0       |
| Skarmory              | 0.00      | 0.00   | 0.00     | 0       |
| Skeledirge            | 0.00      | 0.00   | 0.00     | 0       |
| Slither Wing          | 0.00      | 0.00   | 0.00     | 0       |
| Slowbro               | 0.00      | 0.00   | 0.00     | 0       |
| Spinda                | 0.50      | 1.00   | 0.67     | 1       |
| Steelix               | 0.00      | 0.00   | 0.00     | 0       |
| Sudowoodo             | 0.00      | 0.00   | 0.00     | 0       |
| Swampert              | 0.00      | 0.00   | 0.00     | 0       |
| Sylveon               | 0.00      | 0.00   | 0.00     | 0       |
| Tapu Lele             | 0.00      | 0.00   | 0.00     | 0       |
| Tatsugiri             | 0.00      | 0.00   | 0.00     | 0       |
| Tauros-Paldea-Fire    | 0.00      | 0.00   | 0.00     | 0       |
| Tauros-Paldea-Water   | 0.00      | 0.00   | 0.00     | 0       |
| Ting-Lu               | 0.00      | 0.00   | 0.00     | 0       |
| Tinkaton              | 0.00      | 0.00   | 0.00     | 0       |
| Torkoal               | 0.00      | 0.00   | 0.00     | 0       |
| Tornadus-Therian      | 0.00      | 0.00   | 0.00     | 0       |
| Tsareena              | 0.00      | 0.00   | 0.00     | 0       |
| Tyranitar             | 0.00      | 0.00   | 0.00     | 0       |
| Umbreon               | 0.00      | 0.00   | 0.00     | 0       |
| Urshifu-*             | 0.00      | 0.00   | 0.00     | 0       |
| Vivillon-Fancy        | 0.00      | 0.00   | 0.00     | 0       |
| Volcarona             | 0.00      | 0.00   | 0.00     | 0       |
| Whiscash              | 0.00      | 0.00   | 0.00     | 0       |
| accuracy              |           |        | 0.89     | 3848    |
| macro avg             | 0.58      | 0.53   | 0.53     | 3848    |
| weighted avg          | 0.90      | 0.89   | 0.89     | 3848    |