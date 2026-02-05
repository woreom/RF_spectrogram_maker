import os
import logging
import traceback
import coloredlogs
from constants import SAMPLING_RATE, FFT_SIZE, NUM_FFT_SPEC
from dataset import make_dataset


# Setup colored logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='WARNING', logger=logger, fmt='%(asctime)s %(levelname)s %(message)s')


def main(snr_range,
         dataset_path='../../dataset/unannotated_cage_dataset/',
         base_path='../../dataset/Relate work from AeroDefense/Data/Raw data/SR20M_G50_cage_RT15/',
         patterns=[os.path.join("**", "*.dat")],
         sampling_rate=SAMPLING_RATE,
         sample_time=0.5,
         step_time=0.1,
         fft_size=FFT_SIZE,
         num_fft_spec=NUM_FFT_SPEC,
         json_save_file='meta_data.json',
        add_original=True):

    snr_list = [None] if add_original else []  # Include original data if specified
    snr_list.extend(list(snr_range))
    for snr_db in snr_list:
        logger.info(f"Creating dataset with SNR: {snr_db} dB")
        # Create the dataset path if it doesn't exist
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
            logger.info(f"Created dataset directory: {dataset_path}")
        else:
            logger.info(f"Dataset directory already exists: {dataset_path}")

        snr_dataset_path = os.path.join(dataset_path, f'{snr_db}db' if snr_db is not None else 'original')
        if not os.path.exists(snr_dataset_path):
            os.makedirs(snr_dataset_path)
            logger.info(f"Created SNR dataset directory: {snr_dataset_path}")
        else:
            logger.info(f"SNR dataset directory already exists: {snr_dataset_path}")
        
        # Create the dataset with the specified SNR
        dataset = make_dataset(
            base_path=base_path,
            patterns=patterns,
            dataset_path=snr_dataset_path,
            sampling_rate=sampling_rate,
            sample_time=sample_time,
            step_time=step_time,
            fft_size=fft_size,
            num_fft_spec=num_fft_spec,
            snr_db=snr_db,
            json_save_file=json_save_file
        )

        # no_drone_files = [
        #     '../../data/Relate work from AeroDefense/Data/Raw data/NoDrone_indoor/NoDrone_Band24GHz_Freq2442_SampRate20MHz_ReqTime30sec-002.dat',
        #     '../../data/Relate work from AeroDefense/Data/Raw data/NoDrone_indoor/NoDrone_Band900MHz_Freq918_SampRate20MHz_ReqTime30sec-002.dat',
        #     '../../data/Relate work from AeroDefense/Data/Raw data/NoDrone_outdoor/NoDrone905_SampRate20MHz.dat',
        #     '../../data/Relate work from AeroDefense/Data/Raw data/NoDrone_outdoor/NoDrone2442_SampRate20MHz.dat'
        # ]

        # dataset = make_dataset_for_no_drone(
        #     dat_files=no_drone_files,
        #     dataset_path=snr_dataset_path,
        #     sampling_rate=sampling_rate,
        #     sample_time=sample_time,
        #     step_time=step_time,
        #     fft_size=fft_size,
        #     num_fft_spec=num_fft_spec,
        #     snr_db=snr_db,
        # )

        logger.info(f"Dataset created for {snr_db}db with {len(dataset)} entries.")


# %%
if __name__ == "__main__":
    # Define the SNR range for dataset creation
    # snr_range = range(-20, 21, 5)  # Example: from -20 dB to 20 dB in steps of 5 dB
    import os
    snr_range = []
    base_folder = '2025-12-12'
    dataset_path=f'../../dataset/SiteSurveyRecord/{base_folder}/'
    base_path=f'../../dataset/SiteSurveyRecord/{base_folder}/'
    for folder_name in os.listdir(base_path):
        # Call the main function to create the dataset
        json_output_name = f"site_survey_{base_folder}_{folder_name}.json"

        main(snr_range,
            dataset_path=os.path.join(dataset_path, folder_name),
            base_path=os.path.join(base_path, folder_name), 
            add_original=True)  # Set add_original=True to include the original data without noise
# %%
