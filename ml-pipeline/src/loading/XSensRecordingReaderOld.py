import os
import pandas as pd
import re

from utils.config import Config


class XSensRecordingReaderOld(object):
    """
    XSensRecordingReader.get_recording_frame() concatenates all sensor csv files in the recording folder to one data frame
    """

    @staticmethod
    def get_recording_frame(recording_folder_path) -> pd.DataFrame:
        """
        All files in format 'ID_timestamp.csv' should be in recording_folder_path to get loaded
        """

        # This is the output variable
        recording_frame = None
        # Iterate over all files in the data directory
        for file_name in os.listdir(recording_folder_path):
            is_file = os.path.isfile(os.path.join(recording_folder_path, file_name))
            if not is_file or not file_name.endswith(".csv"):
                continue
            sensor_file_name = file_name

            # Extract the sensor ID from the filename
            # sensor_id = sensor_file_name.split('.')[0].split('_')[0]
            mac_regex = re.compile(r"(?:[0-9a-fA-F]:?){12}", re.IGNORECASE)
            sensor_mac_address = re.findall(mac_regex, sensor_file_name)
            sensor_id = Config.sonar_st1_sensor_map[sensor_mac_address[0]] if len(sensor_mac_address) > 0 else sensor_file_name.split('-')[0].upper()

            # Complete path for reading & read it
            sensor_file_path = os.path.join(
                recording_folder_path, sensor_file_name)
            sensor_frame = pd.read_csv(
                sensor_file_path, skiprows=Config.sonar_csv_header_size)
            # print(f"Adding file {sensor_file_name} with id {sensor_id} and shape {sensor_frame.shape}")

            # Add new frame to recording_frame
            if recording_frame is None:
                # Init the recording_frame with the dataframe, but add the respective suffix
                # but keep SampleTimeFine
                recording_frame = XSensRecordingReaderOld.__prepare_dataframe(
                    sensor_frame, sensor_id)
            else:
                sensor_frame = XSensRecordingReaderOld.__prepare_dataframe(
                    sensor_frame, sensor_id)
                recording_frame = XSensRecordingReaderOld.__merge_frames(
                    recording_frame, sensor_frame)
        recording_frame = XSensRecordingReaderOld.__remove_edge_nans(recording_frame)
        return recording_frame

    @staticmethod
    def __prepare_dataframe(frame, identifier):
        suffix = '_' + identifier
        del frame['PacketCounter']
        # Fill all values of columns that begin with Quat_ with 0
        # frame['Quat_W'] = 0
        # frame['Quat_X'] = 0
        # frame['Quat_Y'] = 0
        # frame['Quat_Z'] = 0
        #del frame['Status']
        return XSensRecordingReaderOld.__add_suffix_except_SampleTimeFine(frame, suffix)

    # Adds a suffix to all columns, but SampleTimeFine
    @staticmethod
    def __add_suffix_except_SampleTimeFine(frame, suffix):
        rename_dictionary = {}
        rename_dictionary['SampleTimeFine'+suffix] = 'SampleTimeFine'
        return frame.add_suffix(suffix).rename(columns=rename_dictionary)

    @staticmethod
    def __merge_frames(frame1, frame2):
        return pd.merge_asof(frame1, frame2, on='SampleTimeFine', tolerance=16000, direction='nearest')
        # return pd.merge(frame1, frame2, on='SampleTimeFine', how='outer')

    @staticmethod
    def __remove_edge_nans(frame):
        rows_before = frame.shape[0]
        frame = XSensRecordingReaderOld.__remove_initial_nans(frame)
        frame = XSensRecordingReaderOld.__remove_ending_nans(frame)
        rows_after = frame.shape[0]

        number_of_rows_removed = rows_before - rows_after
        if number_of_rows_removed > 10:
            print(f"Warning: Removed {number_of_rows_removed} rows from the recording")
            if number_of_rows_removed > 10000:
                raise Exception("This is a very large number of rows, check the data")

        return frame

    @staticmethod
    def __remove_initial_nans(data):
        """
        Removes the first rows where any value is nan.
        """
        removed = 0
        while data.iloc[0].isna().any():
            removed += 1
            data = data.iloc[1:]
            # if removed > 10:
            #     print("More than 10 rows removed")
        return data

    @staticmethod
    def __remove_ending_nans(data):
        """
        Removes the last rows where any value is nan.
        """
        while data.iloc[-1].isna().any():
            data = data.iloc[:-1]
        return data
