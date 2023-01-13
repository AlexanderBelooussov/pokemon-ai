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
    for page in tqdm(range(1, page_limit+1), desc=f'Searching and downloading {format} replays'):
        url = f'https://replay.pokemonshowdown.com/search.json?format={format}&page={page}'
        search = requests.get(url, allow_redirects=True)
        search_json = search.json()
        info.extend(search_json)
        for result in tqdm(search_json, desc='Downloading replays', leave=False):
            if not os.path.exists(f'replays/{format}/{result["id"]}.{type}'):
                replay_url = f"https://replay.pokemonshowdown.com/{result['id']}.{type}"
                replay = requests.get(replay_url, allow_redirects=True)
                content = replay.content
                raw_replays.append(content)
                downloaded += 1
    # now write the results to files
    for i, result in tqdm(enumerate(raw_replays), desc='Writing replays'):
        with open(f'replays/{format}/{info[i]["id"]}.{type}', 'wb') as f:
            f.write(result)

    print(f'Downloaded {downloaded} {format} replays\n')


if __name__ == '__main__':
    search_format()
    search_format(format='gen9monotype')
    search_format(format='gen9vgc2023series1')
    search_format(format='gen9doublesou')
    search_format(format='gen9nationaldex')
    search_format(format='gen9uu')
    search_format(format='gen9ru')
    search_format(format='gen9ubers')
    search_format(format='gen9vgc2023series2')
    time.sleep(5)