import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pillow_heif
from PIL import Image
from tqdm import tqdm


def article(next_word):
    return 'an' if next_word.startswith(('a', 'e', 'i', 'o', 'u')) else 'a'


pillow_heif.register_avif_opener()

df = pd.read_csv('data/Furniture - Text Captions.csv')

df['file_name'] = df['file_name'].map(lambda fn: fn.replace('"', '').replace("'", ""))
df['instance'] = df['text'].map(lambda t: t.lower().replace('/', ' ').replace(':', ''))
df = df.drop(columns='text')

FURNITURE_PATH = Path('data/good_manual')

# Leave only descriptions of existing images
df = df[df.file_name.map(lambda fn: (FURNITURE_PATH / (fn + '.jpg')).exists())]
df = df.reset_index(drop=True)

df['class'] = ''


# Assign class based on the file name
classes = [
    'armchair', 'chair', 'loveseat', 'sectional', 'storage',
    'table', 'couch', 'ottoman', 'chaise', 'sofa'
]
for i, name in enumerate(df.file_name):
    name = name.lower()
    possible_classes = [c for c in classes if c in name]
    if possible_classes:
        df.loc[i, 'class'] = possible_classes[0]
    else:
        if 'mar 100' in name or 'amorie 79' in name:
            df.loc[i, 'class'] = 'sectional'
assert (df['class'] == '').sum() == 0

df['class'][df['class'] == 'armchair'] = 'chair'
df['class'][df['class'] == 'ottoman'] = 'table'
df['class'][df['class'] == 'couch'] = 'sofa'

# Regroup images into folders based on instance (text)
output_dir_instances = Path('data/dataset/instances')
if output_dir_instances.exists():
    shutil.rmtree(output_dir_instances)
output_dir_instances.mkdir()
for instance, group in df.groupby('instance'):
    instance = instance.replace('/', ' ').replace(':', '')
    (output_dir_instances / instance).mkdir()
    for file_name in group['file_name']:
        file_name += '.jpg'
        shutil.copyfile(FURNITURE_PATH / file_name, output_dir_instances / instance / file_name)


# Regroup images into folders based on class (text)
output_dir_classes = Path('data/dataset/classes')
if output_dir_classes.exists():
    shutil.rmtree(output_dir_classes)
output_dir_classes.mkdir()
for instance, group in df.groupby('class'):
    (output_dir_classes / instance).mkdir()
    for file_name in group['file_name']:
        file_name += '.jpg'
        shutil.copyfile(FURNITURE_PATH / file_name, output_dir_classes / instance / file_name)

# We will use living room as one of the instances and classes
living_room_path = Path('data/source_data/Living Room Images')
living_room_path_class = output_dir_instances / 'living room'
living_room_path_instance = output_dir_classes / 'living room'
if living_room_path_instance.exists():
    shutil.rmtree(living_room_path_instance)
if living_room_path_class.exists():
    shutil.rmtree(living_room_path_class)
living_room_path_class.mkdir()
living_room_path_instance.mkdir()
for file_path in living_room_path.glob('*'):
    img = Image.open(file_path)
    img.save(living_room_path_class / (file_path.name + '.jpg'))
    img.save(living_room_path_instance / (file_path.name + '.jpg'))


# generate concepts_list code for ShivamShrirao/DreamBooth_Stable_Diffusion.ipynb
instance_class = df[['instance', 'class']].drop_duplicates(subset='instance')
for _, (instance, class_) in instance_class.iterrows():
    print(
        f'{{\n'
        f'  "instance_prompt": "a white background photo of {article(instance)} {instance}",\n'
        f'  "class_prompt": "a white background photo of {article(instance)} {class_}",\n'
        f'  "instance_data_dir": "/content/furniture/instances/{instance}",\n'
        f'  "class_data_dir": "/content/furniture/classes/{class_}"\n'
        f'}},'
    )

counts = df[['file_name', 'instance']].groupby('instance').nunique()
counts = counts.rename(columns={'file_name': 'count'})
counts = counts.sort_values(by='count')

# Exploratory data analysis
plt.hist(counts, bins='auto')
plt.tight_layout()
plt.show()

plt.hist(counts[counts < 20], bins='auto')
plt.tight_layout()
plt.show()

# Generate test prompts
prompts = df.merge(counts, left_on='instance', right_index=True)
# It sometimes generates too rare instance numbers because they are uniform
# Also we want as different text prompts as possible, so we need to sample several
for _ in range(30):
    best_uniformity = float('-inf')
    best_prompts = None
    best_diff = None
    for i in tqdm(range(5000)):
        # Sample 1 per class
        prompts1 = prompts.groupby(['class']).sample()

        diffs = np.diff(sorted(prompts1['count']))
        # We want to know how different number of instances affects the generation results,
        # so we try to seek a uniform distribution of instance numbers
        uniformity = np.mean(diffs) - np.std(diffs)
        if uniformity > best_uniformity:
            best_prompts = prompts1.copy(deep=True)
            best_uniformity = uniformity
            best_diff = diffs
    best_prompts = best_prompts.sort_values('count')
    print(best_prompts['count'].values)
    print('\n'.join(best_prompts['instance']))
