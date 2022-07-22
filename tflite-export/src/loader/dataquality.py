"""
get recordings from MagentaCloud
1. put the credentials.txt file in the root folder
2. Installations
    - conda install webdavclient3 doesnt work you need to leave the environment conda deativate
    - do it with pip install webdavclient3
3. run the script from root! (python3 src/loader/dataquality.py)

- watch the progress by watching data/sonar-dataset getting more and more folders
- in Lucas personal measurement: it takes 2 eterneties to download all recordings
- you will probably stop the download and only use a subset of the data
"""

from collections import defaultdict
import numpy as np
import utils.settings as settings
from webdav3.client import Client
from loader.XSensRecordingReader import XSensRecordingReader
import pandas as pd
import os
import urllib.request, json
import matplotlib.pyplot as plt
import time

def calculate_acceleration_for_file(filepath) -> None:
    header = ''

    with open(filepath, 'r') as f:
        for i in range(8):
            header += f.readline()

    data = pd.read_csv(filepath, header=7)
    data['acc_x'] = data['dv[1]'] * 60
    data['acc_y'] = data['dv[2]'] * 60
    data['acc_z'] = data['dv[3]'] * 60

    with open(filepath, 'w') as f:
        f.write(header)

    data.to_csv(filepath, index=False, mode="a")

def download_recordings(path):
    options = {}
    with open("credentials.txt", 'r') as f:
        options = {
            'webdav_hostname': f.readline().strip(),
            'webdav_login': f.readline().strip(),
            'webdav_password': f.readline().strip(),
        }
    client = Client(options)
    client.pull(remote_directory="/ML Prototype Recordings", local_directory=path)

def fetch_recordings(local_data_path: str) -> str:
    download_recordings(local_data_path)
    dir_list_after = os.listdir(local_data_path)
    return dir_list_after


def get_nan_of_folder(complete_folder_name, folder_name):
    Nan_tuple_list = []
    df = XSensRecordingReader.get_recording_frame(complete_folder_name)

    nonzero = np.count_nonzero(df.isnull().values, axis=0)
    nonzero_df_with_col_names = pd.DataFrame(
        data=nonzero.reshape(1, -1), columns=df.columns
    )

    prefixes = list(set([column[-17:] for column in df.columns]))
    prefixes.remove("SampleTimeFine")

    columns = ["Quat_W_" + prefix for prefix in prefixes]

    for column in columns:
        Nan_tuple_list.append(
            (
                folder_name,
                column[-17:],
                nonzero_df_with_col_names[column].values[0],
                nonzero_df_with_col_names[column].values[0] * 100 / df.shape[0],
            )
        )
    return Nan_tuple_list


def replace_all_timestamps_minus_one_hour(local_data_path):
    recordings = os.listdir(local_data_path)
    for recording in recordings:
        complete_folder_name = os.path.join(local_data_path, recording)
        # get metadata
        with open(complete_folder_name + "/metadata.json", "r") as f:
            metadata = json.load(f)
        metadata["endTimestamp"] = metadata["endTimestamp"] - 3600000
        metadata["startTimestamp"] = metadata["startTimestamp"] - 3600000
        activities = metadata["activities"]
        for pair in activities:
            pair["timeStarted"] = pair["timeStarted"] - 3600000
        with open(complete_folder_name + "/metadata.json", "w") as f:
            json.dump(metadata, f)


def add_acceleration_to_recording(complete_folder_name):
    files = os.listdir(complete_folder_name)
    files.remove("metadata.json")

    for file in files:
        assert file[-4:] == ".csv", "found unsupported file (not .csv): " + file
        calculate_acceleration_for_file(complete_folder_name + "/" + file)


def merge_retransmissions(complete_folder_name, local_path, folder_name):
    files = os.listdir(complete_folder_name)
    files.remove("metadata.json")

    for file in files:
        header = ""
        with open(complete_folder_name + "/" + file, "r") as f:
            for i in range(8):
                header += f.readline()

        data = pd.read_csv(complete_folder_name + "/" + file, header=7)
        retransmissions = [g for _, g in data.groupby("SampleTimeFine") if len(g) > 1]
        idx_to_remove = []

        for group in retransmissions:
            length = len(group)
            for index, row in group.iterrows():
                zero_count = len([value for value in row if value == 0.0]) - 1
                if zero_count:
                    idx_to_remove.append(index)
                    length -= 1
            if length > 1:
                idx_to_remove.append(group.index[-1])

        if idx_to_remove:
            data = data.drop(idx_to_remove)
            save_path = local_path + "Retransmissions fixed/" + folder_name
            create_dir(save_path)
            data.to_csv(save_path + "/" + file, index=False)


def create_dir(path):
    try:
        os.mkdir(path)
    except OSError as error:
        # print("folder creation not successful: " + str(error))
        print("path exists")


if __name__ == "__main__":

    """
    This looks like the control panel, where you can control, what you want to do by executing python3 dataquality.py
    - for example, I only want to download the data
    """
    fetch = True
    verbose = True
    NaN = False
    distribution = False
    add_acceleration = False
    retransmissions = True
    settings.init()
    local_path = "data/"
    local_data_path = local_path + "sonar-dataset/"
    create_dir(local_path)
    create_dir(local_data_path)


    diff = []
    if fetch:
        diff = fetch_recordings(local_data_path)
    else:
        diff = os.listdir(local_data_path)

    with open(local_path + "labels.json", "r") as f:
        label_items = json.loads(f.read())["items"]
        labels = ["unbekannt", "invalid"]
        for pair in label_items:
            labels.extend(pair["entries"])
    with open(local_path + "people.json", "r") as f:
        persons = json.loads(f.read())["items"]
        persons.remove("unknown")

    new_labels = []
    activity_dict = {}
    activity_dict = defaultdict(lambda: 0, activity_dict)
    if retransmissions:
        create_dir(local_path + "Retransmissions fixed/")

    # substract one hour from all metadata timestamps and write back to metadata.json
    # replace_all_timestamps_minus_one_hour(local_data_path)
    # exit()

    # list of all NaN Values
    # one element is in shape of (recording_folder_name, sensor, amount_of_NaN_values, percentage_of_NaN_values)
    NaN_tuple_list = []

    for folder_name in diff:
        complete_folder_name = os.path.join(local_data_path, folder_name)

        # missing files
        if not len(os.listdir(complete_folder_name)) == 6:
            print("missing files in " + folder_name)
            print(len(os.listdir(complete_folder_name)))
            continue
        elif verbose == True:
            print("number of files in " + folder_name + " is correct")

        # get metadata
        with open(complete_folder_name + "/metadata.json", "r") as f:
            metadata = json.load(f)

        # sensorMap check
        sensorMap_problem = False
        if len(metadata["sensorMapping"].items()) != 5:
            sensorMap_problem = True
        for key, value in metadata["sensorMapping"].items():
            sensorMap_problem = sensorMap_problem or not (len(key) == 17)
            sensorMap_problem = sensorMap_problem or not (len(value) == 4)
        if verbose == True or sensorMap_problem == True:
            print(
                "sensorMap for "
                + folder_name
                + " has problems: "
                + str(sensorMap_problem)
            )

        # label check
        label_problem = False
        activities = metadata["activities"]
        if len(activities) == 0:
            print("labels for " + folder_name + " are empty!!!")
            label_problem = True
        for pair in activities:
            if pair["label"] not in labels and pair["label"] != "invalid":
                if verbose == True:
                    print(
                        'label "'
                        + pair["label"]
                        + '" of '
                        + folder_name
                        + " not in labels"
                    )
                new_labels.append(pair["label"])
                label_problem = True
        if verbose == True or label_problem == True:
            print("labels for " + folder_name + " have problems: " + str(label_problem))

        # person check
        label_problem = False
        person = metadata["person"]
        if person not in persons:
            print("person for " + folder_name + " has problematic input")

        # NaN check
        if NaN:
            temp_list = get_nan_of_folder(complete_folder_name, folder_name)

        # activity distribution in minutes
        if distribution:
            activities = metadata["activities"]
            for idx, pair in enumerate(activities):
                if idx == len(activities) - 1:
                    activity_dict[pair["label"]] += (
                        metadata["endTimestamp"] - pair["timeStarted"]
                    ) / (1000 * 60)
                else:
                    activity_dict[pair["label"]] += (
                        activities[idx + 1]["timeStarted"] - pair["timeStarted"]
                    ) / (1000 * 60)

        # add the accelerations to file
        if add_acceleration:
            add_acceleration_to_recording(complete_folder_name)

        if retransmissions:
            merge_retransmissions(complete_folder_name, local_path, folder_name)

    print("\n".join(list(set(new_labels))))

    if distribution:
        activity_dict = dict(activity_dict)
        non_null_minutes = 0

        # sort and round item to 2 decimal
        activity_dict = dict(
            sorted(activity_dict.items(), key=lambda item: item[1], reverse=True)
        )
        activity_dict = {k: round(v, 2) for k, v in activity_dict.items()}
        for k, v in activity_dict.items():
            if k != "null - activity":
                non_null_minutes += v
        print(
            "recording minutes that are not null activities: " + str(non_null_minutes)
        )
        # filter activities, group all that are not top 11
        filtered_activity_dict = {}
        total = 0
        for idx, (k, v) in enumerate(activity_dict.items()):
            if idx >= 11:
                total += v
            else:
                filtered_activity_dict[k] = v
        filtered_activity_dict["others"] = total

        data = list(filtered_activity_dict.values())
        lbls = list(filtered_activity_dict.keys())
        plt.figure(figsize=(15, 12))
        plt.pie(data, labels=lbls, startangle=90)
        plt.title("Activity Distribution")
        plt.axis("equal")
        plt.savefig("activities_pie_chart.png")
        # plt.show()

        plt.figure(figsize=(15, 12))
        ax = pd.Series(data).plot(kind="bar")
        ax.set_title("Activitiy Distribution")
        # ax.set_xlabel(lbls)
        ax.set_ylabel("Minutes")
        # ax.set_xticklabels("")

        rects = ax.patches
        for idx, (rect, label) in enumerate(zip(rects, lbls)):
            addHeight = 0
            if idx % 2 == 0:
                addHeight = 15
            if idx == 0:
                addHeight = 0
            if idx == 3:
                addHeight = 15
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height + 5 + addHeight,
                label,
                ha="center",
                va="bottom",
            )

        plt.savefig("activities_bar_chart.png")
        # plt.show()

        # print(filtered_activity_dict)

    if NaN:
        df = pd.DataFrame(
            data=NaN_tuple_list,
            columns=["folder", "sensor", "null_amount", "null_percentage"],
        )
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)
        with open("OUTPUT.txt", "w") as f:
            # get avg, max, min NaN per Sensor
            print(
                df[["sensor", "null_amount"]]
                .groupby("sensor")
                .agg([np.average, "max", "sum"]),
                file=f,
            )

            # get avg, max, min NaN per Recording and Sensor
            print(
                df[["folder", "sensor", "null_amount", "null_percentage"]]
                .groupby(["folder", "sensor"])
                .agg([np.average, "max", "sum"]),
                file=f,
            )

            # get max, sum of NaN per folder
            print(
                df[["folder", "null_percentage"]].groupby("folder").agg(["max", "sum"]),
                file=f,
            )

