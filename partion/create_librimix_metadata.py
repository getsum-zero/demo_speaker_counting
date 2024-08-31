import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm

# Global parameters
# eps secures log and division
EPS = 1e-10
# max amplitude in sources and mixtures
MAX_AMP = 0.9
# In LibriSpeech all the sources are at 16K Hz
RATE = 16000
# We will randomize loudness between this range
MIN_LOUDNESS = -33
MAX_LOUDNESS = -25

MIN_SNR = 0
MAX_SNR = 20

NUMBER_OF_TEST = 3000

# A random seed is used for reproducibility
random.seed(72)




# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--librispeech_md_dir', type=str, required=True,
                    help='Path to librispeech metadata directory')
parser.add_argument('--wham_md_dir', type=str, required=True,
                    help='Path to wham metadata directory')
parser.add_argument('--metadata_outdir', type=str, required=True,
                    help='Where librimix metadata files will be stored.')
parser.add_argument('--n_src', type=int, required=True,
                    help='Number of sources desired to create the mixture')



def read_sources(metadata_file, pair, max_src):
    '''Read the speaker's information from the speech metadata'''

    # Read lines corresponding to pair
    sources = [metadata_file.iloc[pair[i]] for i in range(max_src)]
    # Get sources info
    speaker_id_list = [source['speaker_ID'] for source in sources]
    sex_list = [source['sex'] for source in sources]
    length_list = [source['length'] for source in sources]
    path_list = [source['origin_path'] for source in sources]

    id_l = [os.path.split(source['origin_path'])[1].strip('.flac')
            for source in sources]
    mixtures_id = "_".join(id_l)

    # Get the longest and shortest source len
    max_length = max(length_list)
    sources_list = []

    # ========================= !!!! ========================================
    # ================ Left-aligned, may need to be modified later ===========
    # Read the source and compute some info
    for i in range(max_src):
        source = metadata_file.iloc[pair[i]]
        s, _ = sf.read(source['origin_path'], dtype='float32')
        sources_list.append(
            np.pad(s, (0, max_length - len(s)), mode='constant'))

    sources_info = {'mixtures_id': mixtures_id,
                    'speaker_id_list': speaker_id_list, 'sex_list': sex_list,
                    'path_list': path_list}
    return sources_info, sources_list


def add_noise(wham_md_file, pair_noise, sources_list, sources_info):
    # Get the row corresponding to the index
    noise = wham_md_file.loc[pair_noise[0]]
    noise_path = noise['origin_path']

    sources_info['noise_path'] = noise_path

    n, _ = sf.read(noise_path, dtype='float32')
    if len(n.shape) > 1:  n = n[:, 0]

    # Get expected length
    length = len(sources_list[0])
    endpoint = [0, length]

    # Pad if shorter
    if length > len(n):
        sources_list.append(np.pad(n, (0, length - len(n)), mode='constant'))
        endpoint = [0, length]
    # Cut if longer
    else:
        start = random.randint(0, len(n) - length)
        sources_list.append(n[start:start+length])
        endpoint = [start, start+length]

    return sources_info, sources_list, endpoint

def set_utt_pairs(librispeech_md_file, pair_list, max_src):
    ''' Pairing different speech segments '''
    c = 0
    # Index of the rows in the metadata file
    index = list(range(len(librispeech_md_file)))

    # Try to create pairs with different speakers end after 200 fails
    while len(index) >= max_src and c < 200:
        couple = random.sample(index, max_src)
        # Check that speakers are different
        speaker_list = set([librispeech_md_file.iloc[couple[i]]['speaker_ID']
                            for i in range(max_src)])
     

        if len(speaker_list) != max_src:
            c += 1
        else:
            for i in range(max_src):
                index.remove(couple[i])
            pair_list.append(couple)
            c = 0
    return pair_list

def set_noise_pairs(pairs, noise_pairs, librispeech_md_file, noise_md_file):

    print('Generating pairs')
    md = noise_md_file[noise_md_file['augmented'] == False]

    # If there are more mixtures than noises then use augmented data
    if len(pairs) > len(md):
        md = noise_md_file


    for pair in tqdm(pairs.copy(), total=len(pairs.copy())):
        sources = [librispeech_md_file.iloc[pair[i]]
                   for i in range(len(pair))]
        
        # get max_length
        length_list = [source['length'] for source in sources]
        max_length = max(length_list)
        # Ideal choices are noises longer than max_length
        possible = md[md['length'] >= max_length]

        # if possible is not empty
        try:
            pair_noise = random.sample(list(possible.index), 1)
            noise_pairs.append(pair_noise)

            # ======================== !!! ====================
            # remove that noise from the remaining noises
            # if typedata[idex] == "wham_noise":
            #     md = md.drop(pair_noise)
            # =================================================

        # if possible is empty
        except ValueError:
            # if we deal with training files
            if 'train' in librispeech_md_file.iloc[0]['subset']:
                pair_noise = list(md.index)[-1]
                noise_pairs.append(pair_noise)
            else:
                pairs.remove(pair)

    return noise_pairs

def set_pairs(librispeech_md_file, wham_md_file, max_src):
    """ set pairs of sources to make the mixture """

    utt_pairs = []
    noise_pairs = []

    print("subset:", librispeech_md_file.iloc[0]['subset'])
    # In train sets utterance are only used once
    if 'train' in librispeech_md_file.iloc[0]['subset']:
        utt_pairs = set_utt_pairs(librispeech_md_file, utt_pairs, max_src)
        noise_pairs = set_noise_pairs(utt_pairs, noise_pairs,
                                      librispeech_md_file, wham_md_file)
    
    # Otherwise we want 3000 mixtures
    else:
        while len(utt_pairs) < NUMBER_OF_TEST:
            utt_pairs = set_utt_pairs(librispeech_md_file, utt_pairs, max_src)
            noise_pairs = set_noise_pairs(utt_pairs, noise_pairs,
                                          librispeech_md_file, wham_md_file)
            if max_src > 1:
                utt_pairs, noise_pairs = remove_duplicates(utt_pairs, noise_pairs)

        utt_pairs = utt_pairs[:NUMBER_OF_TEST]
        noise_pairs = noise_pairs[:NUMBER_OF_TEST]

    return utt_pairs, noise_pairs



def mix(sources_list_norm):
    """ Do the mixture for min mode and max mode """
    # Initialize mixture
    mixture_max = np.zeros_like(sources_list_norm[0])
    for i in range(len(sources_list_norm)):
        mixture_max += sources_list_norm[i]
    return mixture_max


def get_row(sources_info, gain_list, max_src, noise_endpoint, snr):
    """ Get new row for each mixture/info dataframe """
    row_mixture = [sources_info['mixtures_id']]
    row_info = [sources_info['mixtures_id']]
    for i in range(max_src):
        row_mixture.append(sources_info['path_list'][i])
        row_mixture.append(gain_list[i])
        row_info.append(sources_info['speaker_id_list'][i])
        row_info.append(sources_info['sex_list'][i])

    row_mixture.append(sources_info['noise_path'])
    row_mixture.append(gain_list[-1])
    row_mixture.append(noise_endpoint[0])
    row_mixture.append(noise_endpoint[1])
    row_mixture.append(snr)
     
    return row_mixture, row_info


def set_snr(sources_list_norm, max_src):
    """ Set SNR between sources """
    snr = random.uniform(MIN_SNR, MAX_SNR)

    power_noise = np.sum(np.square(sources_list_norm[-1]))
    clean_mix = np.sum(sources_list_norm[:max_src], axis=0)
    power_mixture =  np.sum(np.square(clean_mix))

    scaling = np.sqrt(power_mixture / (power_noise * np.power(10, snr / 10)))
    sources_list_norm[-1] *= scaling
    return snr, sources_list_norm


def create_librimix_df(librispeech_md_file, wham_md_file, max_src):
    """ Generate librimix dataframe from a LibriSpeech and wha md file"""
    # mixtures_md  &  mixtures_info
    mixtures_md = pd.DataFrame(columns=['mixture_ID'])
    mixtures_info = pd.DataFrame(columns=['mixture_ID'])


    # Add columns (depends on the number of sources)
    for i in range(max_src):
        mixtures_md[f"source_{i + 1}_path"] = {}
        mixtures_md[f"source_{i + 1}_gain"] = {}
        mixtures_info[f"speaker_{i + 1}_ID"] = {}
        mixtures_info[f"speaker_{i + 1}_sex"] = {}

    mixtures_md["noise_path"] = {}
    mixtures_md["noise_gain"] = {}
    mixtures_md["noise_l_endpoint"] = {}
    mixtures_md["noise_r_endpoint"] = {}
    mixtures_md["SNR"] = {}


    # Generate pairs of sources to mix
    pairs, pairs_noise = set_pairs(librispeech_md_file, wham_md_file, max_src)


    clip_counter = 0
    # For each combination create a new line in the dataframe
    print("Generate metadata based on speech pairs:")
    for pair, pair_noise in tqdm(zip(pairs, pairs_noise), total=len(pairs)):

        # ======================= return infos about the sources and calculate gain ===================
        # Left-aligned speech, zero padding for missing speech; used to calculate gain
        sources_info, sources_list_max = read_sources(librispeech_md_file, pair, max_src)
        

        # ======================= Add noise ===================
        sources_info, sources_list_max, noise_endpoint = add_noise(
            wham_md_file, pair_noise, sources_list_max, sources_info)
        
        # compute initial loudness, randomize loudness and normalize sources
        loudness, _, sources_list_norm = set_loudness(sources_list_max)

        snr, sources_list_norm = set_snr(sources_list_norm, max_src)
       

        #  ================ Do the mixture, add =====================
        mixture_max = mix(sources_list_norm)

        # Check the mixture for clipping and renormalize if necessary
        # 使用语音混合后是否造成响度变换，如果有则重新归一化
        # 这里实际并没有进行混合，只是计算了混合后的响度
        renormalize_loudness, did_clip = check_for_cliping(mixture_max,
                                                           sources_list_norm)
        clip_counter += int(did_clip)
        # Compute gain
        gain_list = compute_gain(loudness, renormalize_loudness)

        # Add information to the dataframe
        row_mixture, row_info = get_row(sources_info, gain_list, max_src, noise_endpoint, snr)
        mixtures_md.loc[len(mixtures_md)] = row_mixture
        mixtures_info.loc[len(mixtures_info)] = row_info

    print(f"Among {len(mixtures_md)} mixtures, {clip_counter} clipped.")
    return mixtures_md, mixtures_info



def create_librimix_metadata(librispeech_md_dir, noise_md_dir, md_dir, max_src):
    """ Generate LibriMix metadata according to LibriSpeech metadata """

    # folder to files
    librispeech_md_files = os.listdir(librispeech_md_dir)
    noise_md_files = os.listdir(noise_md_dir)


    # Go through each metadata file and create metadata accordingly
    # 枚举每一个file
    for librispeech_md_file in librispeech_md_files:
        if not librispeech_md_file.endswith('.csv'):
            print(f"{librispeech_md_file} is not a csv file, continue.")
            continue

        # Get the name of the corresponding noise md file
        # file start with "test", "dev", "train"
        try:
            noise_md_file = [f for f in noise_md_files if
                            f.startswith(librispeech_md_file.split('-')[0])][0]
        except IndexError:
            print('Wham metadata are missing you can either generate the '
                  'missing wham files or add the librispeech metadata to '
                  'to_be_ignored list')
            break
        

        
        # =================  Store Filenames ==========================
        # eg. [librimix]_[train-clean-100].csv
        # [librimix]_[train-clean-100]_info.csv
        dataset = "librimix"
        save_path = os.path.join(md_dir,
                                 '_'.join([dataset, librispeech_md_file]))
        info_save_path = os.path.join(md_dir, '_'.join([dataset, librispeech_md_file.strip('.csv'),
                              'info']) + '.csv')
        print(f"Creating {os.path.basename(save_path)} file in {md_dir}")
        

        # =================  Open .csv files ========================
        librispeech_md = pd.read_csv(os.path.join(
            librispeech_md_dir, librispeech_md_file), engine='python')
        noise_md = pd.read_csv(os.path.join(
            noise_md_dir, noise_md_file), engine='python')


        # =================  Create dataframe ========================
        mixtures_md, mixtures_info = create_librimix_df(
            librispeech_md, noise_md,  max_src)
    
        mixtures_md = mixtures_md[:len(mixtures_md) // 100 * 100]
        mixtures_info = mixtures_info[:len(mixtures_info) // 100 * 100]


        # =================      save   ========================
        mixtures_md.to_csv(save_path, index=False)
        mixtures_info.to_csv(info_save_path, index=False)


def main(args):
    librispeech_md_dir = args.librispeech_md_dir
    noise_md_dir = args.wham_md_dir
    max_src = args.n_src

    # Create Librimix metadata directory
    md_dir = args.metadata_outdir
    os.makedirs(md_dir, exist_ok=True)

    create_librimix_metadata(librispeech_md_dir, noise_md_dir, md_dir, max_src)














def remove_duplicates(utt_pairs, noise_pairs):
    print('Removing duplicates')
    # look for identical mixtures O(n²)
    for i, (pair, pair_noise) in enumerate(zip(utt_pairs, noise_pairs)):
        for j, (du_pair, du_pair_noise) in enumerate(
                zip(utt_pairs, noise_pairs)):
            # sort because [s1,s2] = [s2,s1]
            if sorted(pair) == sorted(du_pair) and i != j:
                utt_pairs.remove(du_pair)
                noise_pairs.remove(du_pair_noise)
    return utt_pairs, noise_pairs







def set_loudness(sources_list):
    """ Compute original loudness and normalise them randomly """
    # Initialize loudness
    loudness_list = []
    # In LibriSpeech all sources are at 16KHz hence the meter
    meter = pyln.Meter(RATE)
    # Randomize sources loudness
    target_loudness_list = []
    sources_list_norm = []

    # Normalize loudness
    for i in range(len(sources_list)):
        # Compute initial loudness
        loudness_list.append(meter.integrated_loudness(sources_list[i]))
        # Pick a random loudness
        target_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
        # Noise has a different loudness
        if i == len(sources_list) - 1:
            target_loudness = random.uniform(MIN_LOUDNESS - 5,
                                             MAX_LOUDNESS - 5)
        # Normalize source to target loudness

        with warnings.catch_warnings():
            # We don't want to pollute stdout, but we don't want to ignore
            # other warnings.
            warnings.simplefilter("ignore")
            src = pyln.normalize.loudness(sources_list[i], loudness_list[i],
                                          target_loudness)
        # If source clips, renormalize
        if np.max(np.abs(src)) >= 1:
            src = sources_list[i] * MAX_AMP / np.max(np.abs(sources_list[i]))
            target_loudness = meter.integrated_loudness(src)

        # Save scaled source and loudness.
        sources_list_norm.append(src)
        target_loudness_list.append(target_loudness)

    return loudness_list, target_loudness_list, sources_list_norm





def check_for_cliping(mixture_max, sources_list_norm):
    """Check the mixture (mode max) for clipping and re normalize if needed."""
    # Initialize renormalized sources and loudness
    renormalize_loudness = []
    clip = False
    # Recreate the meter
    meter = pyln.Meter(RATE)
    # Check for clipping in mixtures
    if np.max(np.abs(mixture_max)) > MAX_AMP:
        clip = True
        weight = MAX_AMP / np.max(np.abs(mixture_max))
    else:
        weight = 1
    # Renormalize
    for i in range(len(sources_list_norm)):
        new_loudness = meter.integrated_loudness(sources_list_norm[i] * weight)
        renormalize_loudness.append(new_loudness)
    return renormalize_loudness, clip


def compute_gain(loudness, renormalize_loudness):
    """ Compute the gain between the original and target loudness"""
    gain = []
    for i in range(len(loudness)):
        delta_loudness = renormalize_loudness[i] - loudness[i]
        gain.append(np.power(10.0, delta_loudness / 20.0))
    return gain




if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
