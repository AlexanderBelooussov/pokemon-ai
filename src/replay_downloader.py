import os
import time

import requests
from tqdm import tqdm


# TODO: make index file

def search_format(format='gen9ou', page_limit=25):
    os.makedirs(f'replays/{format}', exist_ok=True)
    raw_replays = []
    info = []
    type = "log"
    downloaded = 0
    done = False
    for page in tqdm(range(1, page_limit + 1), desc=f'Searching and downloading {format} replays'):
        if done:
            break
        url = f'https://replay.pokemonshowdown.com/search.json?format={format}&page={page}'
        while True:
            try:
                search = requests.get(url, allow_redirects=True)
                search_json = search.json()
                break
            except:
                # print('Error, retrying...')
                time.sleep(5)
        info.extend(search_json)
        for result in tqdm(search_json, desc='Downloading replays', leave=False):
            if not os.path.exists(f'replays/{format}/{result["id"]}.{type}'):
                replay_url = f"https://replay.pokemonshowdown.com/{result['id']}.{type}"
                while True:
                    try:
                        replay = requests.get(replay_url, allow_redirects=True)
                        content = replay.content
                        break
                    except:
                        # print('Error downloading replay, retrying...')
                        time.sleep(5)
                raw_replays.append(content)
                downloaded += 1
            else:
                done = True
                break

    # now write the results to files
    for i, result in tqdm(enumerate(raw_replays), desc='Writing replays', leave=False):
        with open(f'replays/{format}/{info[i]["id"]}.{type}', 'wb') as f:
            f.write(result)

    print(f'Downloaded {downloaded} {format} replays\n')


def total_replays():
    total = 0
    replay_counts = {}
    for format in os.listdir('replays'):
        l = len(os.listdir(f'replays/{format}'))
        replay_counts[format] = l
        total += l
    for format, count in sorted(replay_counts.items(), key=lambda x: x[1], reverse=True):
        print(f'{format.ljust(25)}: {count}')
    print(f"\n{'Total replays'.ljust(25)}: {total}\n")


if __name__ == '__main__':
    search_format()
    search_format(format='gen9vgc2023regulationc')
    search_format(format='gen9monotype')
    search_format(format='gen9vgc2023series2')
    search_format(format='gen9nationaldex')
    search_format(format='gen9doublesou')
    search_format(format='gen9nationaldexmonotype')
    search_format(format='gen9battlestadiumsinglesregulationc')
    search_format(format='gen9uu')
    search_format(format='gen9ru')
    search_format(format='gen9ubers')
    search_format(format='gen9nationaldexuu')
    search_format(format='gen9doublesubers')
    search_format(format='gen9doublesuu')
    search_format(format='gen91v1')
    search_format(format='gen92v2doubles')
    search_format(format='gen9lc')
    total_replays()
    input('Press enter to exit')
