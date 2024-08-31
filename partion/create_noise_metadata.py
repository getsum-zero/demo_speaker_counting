import os
import argparse
import soundfile as sf
import pandas as pd
import glob
from tqdm import tqdm
# Global parameter
# We will filter out files shorter than that
NUMBER_OF_SECONDS = 3
# In WHAM! all the sources are at 16K Hz
RATE = 16000

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--wham_dir', type=str, required=True,
                    help='Path to wham_noise root directory')
parser.add_argument('--reproduce', type=bool, default=True,
                    help='Need to reproduce the metadata files or not')
parser.add_argument('--wham_md_dir', type=str, required=True,
                    help='Path to wham_noise md directory')

def main(args):
    wham_noise_dir = args.wham_dir
    reproduce = args.reproduce
    # Create wham_noise metadata directory
    wham_noise_md_dir = args.wham_md_dir
    os.makedirs(wham_noise_md_dir, exist_ok=True)
    create_wham_noise_metadata(wham_noise_dir, wham_noise_md_dir, reproduce)


def create_wham_noise_metadata(wham_noise_dir, md_dir, reproduce):
    """ Generate metadata corresponding to downloaded data in wham_noise """

    # Check already generated files
    # 文件是否已经生成
    not_already_processed_dir = check_already_generated(md_dir, reproduce)
    # Go through each directory and create associated metadata
    for ldir in not_already_processed_dir:
        # Generate the dataframe relative to the directory
        dir_metadata = create_wham_noise_dataframe(wham_noise_dir, ldir)
        # Sort the dataframe according to ascending Length
        dir_metadata = dir_metadata.sort_values('length')
        # Write the dataframe in a .csv in the metadata directory
        if ldir == 'tt':
            name = 'test'
        elif ldir == 'cv':
            name = 'dev'
        else:
            name = 'train'
        # Filter out files that are shorter than 3s
        num_samples = NUMBER_OF_SECONDS * RATE
        dir_metadata = dir_metadata[
            dir_metadata['length'] >= num_samples]
        
        # Create save path
        save_path = os.path.join(md_dir, name + '.csv')
        print(f'Medatada file created in {save_path}')
        dir_metadata.to_csv(save_path, index=False)


def check_already_generated(md_dir, reproduce):
    """ Check if files have already been generated """
    # Get the already generated files
    # 将文件下的文件名存储在列表中
    already_generated_csv = os.listdir(md_dir)

    # Data directories in wham_noise
    wham_noise_dirs = ['cv', 'tr', 'tt']
    # Save the already data directories names
    already_processed_dir = [
        f.replace("test", "tt").replace("train", "tr").replace("dev", "cv").
            replace(".csv", "") for f in already_generated_csv]
    # Actual directories that haven't already been processed
    # 两个列表的差集，得到未处理的部分
    if reproduce:
        not_already_processed_dir = wham_noise_dirs
    else:
        not_already_processed_dir = list(set(wham_noise_dirs) -
                                        set(already_processed_dir))
    return not_already_processed_dir


def create_wham_noise_dataframe(wham_noise_dir, subdir):
    """ Generate a dataframe that gather infos about the sound files in a
    wham_noise subdirectory """

    print(f"Processing files from {subdir} dir")
    # Get the current directory path
    # dir_path = os.path.join(wham_noise_dir, subdir)
    # Recursively look for .wav files in current directory
    sound_paths = glob.glob(os.path.join(wham_noise_dir, '*/{}/*.wav'.format(subdir)),
                            recursive=True)
    
    # Create the dataframe corresponding to this directory
    dir_md = pd.DataFrame(columns=['noise_ID', 'type', 'subset', 'length', 'augmented',
                                   'origin_path'])

    # Go through the sound file list
    for sound_path in tqdm(sound_paths, total=len(sound_paths)):
        # Get the ID of the noise file
        # type / subdir / wav
        dir, noise_id = os.path.split(sound_path)
        type = dir.split('/')[-2]
        # Get its length
        length = len(sf.SoundFile(sound_path))
        augment = False

        # 这里区分的是训练集，训练集数据需要增强
        if 'sp08' in sound_path or 'sp12' in sound_path:
            augment = True

        # Get the sound file relative path
        abs_path = os.path.abspath(sound_path)
        # Add information to the dataframe
        dir_md.loc[len(dir_md)] = [noise_id, type, subdir, length, augment, abs_path]
    return dir_md


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
