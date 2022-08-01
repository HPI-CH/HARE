import os
import pandas as pd
import re
import json
import utils.settings as settings
import numpy as np

verbose = False

if verbose:

    def d_print(*args):
        print(*args)


else:

    def d_print(*args):
        pass


class XSensRecordingReader(object):
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
            mac_regex = re.compile(r"(?:[0-9a-fA-F]-?){12}", re.IGNORECASE)
            sensor_mac_address = re.findall(mac_regex, sensor_file_name)[0]
            with open(recording_folder_path + os.path.sep + "metadata.json", "r") as f:
                data = json.load(f)
            sensor_map_suffix_map = data["sensorMapping"]

            # map from mac address to placements (LW, RW, etc.) without sensor set information
            sensor_id = sensor_map_suffix_map[sensor_mac_address.replace("-", ":")][
                :2
            ].upper()

            # Complete path for reading & read it
            sensor_file_path = os.path.join(recording_folder_path, sensor_file_name)
            sensor_frame = pd.read_csv(
                sensor_file_path, skiprows=settings.DATA_CONFIG.csv_header_size
            )

            # Add new frame to recording_frame
            if recording_frame is None:
                # Init the recording_frame with the dataframe, but add the respective suffix
                # but keep SampleTimeFine
                recording_frame = XSensRecordingReader.__prepare_dataframe(
                    sensor_frame, sensor_id
                )
            else:
                sensor_frame = XSensRecordingReader.__prepare_dataframe(
                    sensor_frame, sensor_id
                )
                recording_frame = XSensRecordingReader.__merge_frames(
                    recording_frame, sensor_frame
                )
        recording_frame = XSensRecordingReader.__remove_edge_nans(recording_frame)
        return recording_frame

    @staticmethod
    def __prepare_dataframe(frame, identifier):
        suffix = "_" + identifier
        if "PacketCounter" in frame.columns:
            del frame["PacketCounter"]
        if "Status" in frame.columns:
            del frame["Status"]
        # Convert all frame values to numbers (otherwise nans might not be read correctly!)
        frame = frame.apply(pd.to_numeric, errors='coerce').astype({"SampleTimeFine": "int64"})
        frame = XSensRecordingReader.__remove_SampleTimeFine_overflow(frame)
        return XSensRecordingReader.__add_suffix_except_SampleTimeFine(frame, suffix)

    @staticmethod
    def __remove_SampleTimeFine_overflow(frame):
        vals = frame["SampleTimeFine"].values
        for idx in range(1, vals.size):
            while vals[idx] < vals[idx - 1]:
                vals[idx] += pow(2, 32) 
        frame["SampleTimeFine"] = vals
        return frame

    # Adds a suffix to all columns, but SampleTimeFine
    @staticmethod
    def __add_suffix_except_SampleTimeFine(frame, suffix):
        rename_dictionary = {}
        rename_dictionary["SampleTimeFine" + suffix] = "SampleTimeFine"
        return frame.add_suffix(suffix).rename(columns=rename_dictionary)

    def __merge_frames(frame1, frame2):
        frame2 = XSensRecordingReader.__normalize_SampleTimeFine(frame1, frame2)
        
        # join on frame1
        df1_2 = pd.merge_asof(frame1, frame2, on='SampleTimeFine', tolerance=16000, direction='nearest')
        
        # join on frame2
        df2_1 = pd.merge_asof(frame2, frame1, on='SampleTimeFine', tolerance=16000, direction='nearest')

        # concat frame1-join with frame2-joins where the last column values (those should always be from frame1) are NaN (equivalent to full outer join)
        df_outer_joined = pd.concat([df1_2,df2_1[df2_1.iloc[:, -1].isnull()]])
        df_outer_joined.sort_values(["SampleTimeFine"], inplace=True)
        return df_outer_joined

    @staticmethod
    def __normalize_SampleTimeFine(frame1, frame2):
        frame1_first_SampleTimeFine = frame1.iloc[0]['SampleTimeFine']
        frame2_first_SampleTimeFine = frame2.iloc[0]['SampleTimeFine']
        diff = frame2_first_SampleTimeFine - frame1_first_SampleTimeFine
        frame2_SampleTimeFine = frame2['SampleTimeFine'].values
        diff_calc = lambda t: t - diff
        frame2_SampleTimeFine = diff_calc(frame2_SampleTimeFine).astype(np.int64)
        frame2['SampleTimeFine'] = frame2_SampleTimeFine
        return frame2

    @staticmethod
    def __remove_edge_nans(frame):
        rows_before = frame.shape[0]
        frame = XSensRecordingReader.__remove_initial_nans(frame)
        frame = XSensRecordingReader.__remove_ending_nans(frame)
        rows_after = frame.shape[0]

        number_of_rows_removed = rows_before - rows_after
        if number_of_rows_removed > 10:
            d_print(
                f"Warning: Removed {number_of_rows_removed} rows from the recording"
            )
            if number_of_rows_removed > 10000:
                d_print("This is a very large number of rows, check the data")

        return frame

    @staticmethod
    def __remove_initial_nans(data):
        """
        Removes the first rows where any value is nan.
        """
        removed = 0
        while data.shape[0] > 0 and data.iloc[0].isna().any():
            removed += 1
            data = data.iloc[1:]
            if removed > 10:
                d_print("More than 10 rows removed")
        return data

    @staticmethod
    def __remove_ending_nans(data):
        """
        Removes the last rows where any value is nan.
        """
        while data.shape[0] > 0 and data.iloc[-1].isna().any():
            data = data.iloc[:-1]
        return data
