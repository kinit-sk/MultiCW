import pandas as pd
from os.path import join
from tqdm.notebook import tqdm
from topic import extract_topics
from translation import init_splitter, cooldown, deeptranslate, parallel_deeptranslate

# ANSI Highlighting: https://stackoverflow.com/a/21786287
h_red = '\x1b[1;30;41m'
h_green = '\x1b[1;30;42m'
h_yellow = '\x1b[1;30;43m'
h_stop = '\x1b[0m'

multicw = pd.read_csv(join('Final-dataset', 'multicw-out.csv'))

# Extract the samples without English translations (e.g. text_en is None)
to_translate = multicw[~(multicw['text_en'].notna() & (multicw['text_en'].str.strip() != ''))].copy()
no_translate = multicw[multicw['text_en'].notna() & (multicw['text_en'].str.strip() != '')].copy()

translated = []
chunk_size = 500
for lang in to_translate['lang'].unique():
    # Select subset for the current language
    subset = to_translate[to_translate['lang'] == lang].copy()

    # Process the subset in chunks
    print(f'{h_green}Processing {lang}:{h_stop}')
    for start_idx in range(0, len(subset), chunk_size):
        end_idx = start_idx + chunk_size
        chunk = subset.iloc[start_idx:end_idx].copy()

        # # Translate original language samples to English
        # print(f'Translating {start_idx} to {end_idx} samples for language {lang}:')
        # chunk.loc[:, 'text_en'] = parallel_deeptranslate(chunk['text'], 'zh-CN' if lang == 'zh' else lang, 'en', max_workers=4)

        # Extract topics from the English translations
        print(f'Extracting topics for {start_idx} to {end_idx} samples for language {lang}:')
        chunk['topic'] = extract_topics(chunk['text_en'])


        translated.append(chunk)

    multicw = pd.concat([no_translate, pd.concat(translated)])
    multicw.to_csv(join('Final-dataset', 'multicw-full.csv'), index=False, header=True)

print(f'Final dataset saved to: multicw-full.csv')