#!/bin/bash
set -eu  # Exit on error






storage_dir=$1   # Path to store the downloaded data


set -eu  # Exit on error
# function LibriSpeech_dev_clean() {
# 	if ! test -e $librispeech_dir/dev-clean; then
# 		echo "Download LibriSpeech/dev-clean into $storage_dir"
# 		# If downloading stalls for more than 20s, relaunch from previous state.
# 		# wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/dev-clean.tar.gz -P $storage_dir
# 		tar -xzf $storage_dir/dev-clean.tar.gz -C $storage_dir
# 		# rm -rf $storage_dir/dev-clean.tar.gz
# 	fi
# }

# function LibriSpeech_test_clean() {
# 	if ! test -e $librispeech_dir/test-clean; then
# 		echo "Download LibriSpeech/test-clean into $storage_dir"
# 		# If downloading stalls for more than 20s, relaunch from previous state.
# 		# wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/test-clean.tar.gz -P $storage_dir
# 		tar -xzf $storage_dir/test-clean.tar.gz -C $storage_dir
# 		# rm -rf $storage_dir/test-clean.tar.gz
# 	fi
# }

# function LibriSpeech_clean100() {
# 	if ! test -e $librispeech_dir/train-clean-100; then
# 		echo "Download LibriSpeech/train-clean-100 into $storage_dir"
# 		# If downloading stalls for more than 20s, relaunch from previous state.
# 		# wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/train-clean-100.tar.gz -P $storage_dir
# 		tar -xzf $storage_dir/train-clean-100.tar.gz -C $storage_dir
# 		# rm -rf $storage_dir/train-clean-100.tar.gz
# 	fi
# }

# function LibriSpeech_clean360() {
# 	if ! test -e $librispeech_dir/train-clean-360; then
# 		echo "Download LibriSpeech/train-clean-360 into $storage_dir"
# 		# If downloading stalls for more than 20s, relaunch from previous state.
# 		# wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/train-clean-360.tar.gz -P $storage_dir
# 		tar -xzf $storage_dir/train-clean-360.tar.gz -C $storage_dir
# 		# rm -rf $storage_dir/train-clean-360.tar.gz
# 	fi
# }

# function wham() {
# 	if ! test -e $wham_dir; then
# 		echo "Download wham_noise into $storage_dir"
# 		# If downloading stalls for more than 20s, relaunch from previous state.
# 		# wget -c --tries=0 --read-timeout=20 https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip -P $storage_dir
# 		unzip -qn $storage_dir/wham_noise.zip -d $storage_dir
# 		# rm -rf $storage_dir/wham_noise.zip
# 	fi
# }

# LibriSpeech_dev_clean
# LibriSpeech_test_clean
# LibriSpeech_clean100
# LibriSpeech_clean360 
# wham 




librispeech_dir=$storage_dir/LibriSpeech
librispeech_md_dir=metadata/LibriSpeech

noise_dir=$storage_dir/noise
noise_md_dir=metadata/noise


echo "preparing metadata of librispeech from $librispeech_dir"
python create_librispeech_metadata.py --librispeech_dir $librispeech_dir \
  --librispeech_md_dir $librispeech_md_dir


echo "preparing metadata of noise dataset from $noise_dir"
python create_noise_metadata.py --wham_dir $noise_dir \
  --wham_md_dir $noise_md_dir


max_src=3
metadata_dir=metadata/LibriMix

echo "preparing metadata of librimix in $noise_dir"
python create_librimix_metadata.py --librispeech_md_dir $librispeech_md_dir \
  --wham_md_dir $noise_md_dir \
  --metadata_outdir $metadata_dir \
  --n_src $max_src


echo "Mix data with $metadata_dir"
librimix_outdir=../data/LibriMix
python create_dataset_from_metadata.py  --metadata_dir $metadata_dir \
	--librimix_outdir $librimix_outdir \
	--n_src $max_src \
	--freqs 16k \
	--modes min \
	--types mix_both




