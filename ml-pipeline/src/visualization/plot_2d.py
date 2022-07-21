from math import sqrt

import matplotlib.pyplot as plt

from utils.config import Config


def plot_freeAccs_graph(recording, columns=None, useMagnitude=False, fixedValueRange=True, plotZeroLine=True,
                        frequency=60, relResolution=(1, 1)):
    """
    Plots specified columns as 2D graph for each sensor. \n
    [columns] must contain valid column names without the sensor suffix (e.g. 'FreeAcc_X' instead of 'FreeAcc_X_LF'). \n
    If [useMagnitude] is set 'True', the euclidean magnitude (absolute value) of all specified [columns] together is shown instead. \n
    With [fixedValueRange] the range of value can be set fixed or individually per sensor plot. \n
    With [relResolution] the resulting .png file can be scaled up or down by the specified factor (width, height). \n
    """

    if columns is None:
        columns = ['FreeAcc_X', 'FreeAcc_Y', 'FreeAcc_Z']
    sensors = Config.sonar_sensor_suffix_order
    numSensors = len(sensors)

    # recordings = load_dataset(os.path.join(settings.DATA_PATH, '5-sensor-all'))
    sensor_frame = recording.sensor_frame
    # print(recordings[recordingID].activity)

    # check if all column names in [column] are valid
    cols_to_plot = list(filter(lambda col: _columnNameMatches(col, columns), sensor_frame.columns))
    if len(cols_to_plot) != len(columns) * numSensors:
        raise Exception("Column names invalid.")

    # specify filename extensions
    attr_suffix1 = "-freeAcc" if _columnNameMatches('FreeAcc_X FreeAcc_Y FreeAcc_Z', columns) else ""
    attr_suffix2 = "-quat" if _columnNameMatches('Quat_W Quat_X Quat_Y Quat_Z', columns) else ""
    mag_suffix = "-mag" if useMagnitude else ""

    # optionally create a magnitude column for each sensor
    if useMagnitude:
        cols_to_plot = []
        for sensor in sensors:
            sensor_cols = sensor_frame.loc[:, [f"{col}_{sensor}" for col in columns]]
            sensor_frame[f'Mag_{sensor}'] = sensor_cols.apply(lambda row: _calcMagnitude(row), axis=1)

            cols_to_plot.append(f'Mag_{sensor}')
        legend_labels = [f'Magnitude {str(columns)}']
        columns = ['Mag']
    else:
        legend_labels = columns

    # calculate highest absolute value of whole data frame
    maxValue = max([sensor_frame[col].max() for col in cols_to_plot])
    minValue = min([sensor_frame[col].min() for col in cols_to_plot])
    maxAbsValue = max(maxValue, abs(minValue))

    numLines = sensor_frame.shape[0]

    fig, axs = plt.subplots(numSensors)
    for index, sensor in enumerate(sensors):
        # extract relevant data frames
        sensor_cols = sensor_frame.loc[:, [f"{col}_{sensor}" for col in columns]]

        # create plot
        ax = axs[index]
        figsize = (int(relResolution[0] * numLines / 35), relResolution[1] * 7)

        # Plot velocity
        # sensor_cols.loc[0, 'velocity'] = (-2) + sensor_cols.iloc[0, 0] * 0.016
        # for i in range(1, len(sensor_cols)):
        #     sensor_cols.loc[i, 'velocity'] = sensor_cols.loc[i-1, 'velocity'] + (sensor_cols.iloc[i, 0] * 0.016)

        # sensor_cols.loc[:, 'velocity'].plot(ax = ax, kind='line', linewidth=0.4*relResolution[1], figsize=figsize)

        sensor_cols.plot(ax=ax, kind='line', linewidth=0.4 * relResolution[1], figsize=figsize)

        # modify plots design
        if plotZeroLine:
            ax.plot([0 for x in range(0, numLines)], '--', linewidth=0.4 * relResolution[1], color='black')
        if fixedValueRange:
            lower_limit = -(1.1 * maxAbsValue) if not useMagnitude else minValue / 1.1
            higher_limit = 1.1 * maxAbsValue if not useMagnitude else maxValue * 1.1
            ax.set_ylim([lower_limit, higher_limit])
        # ax.get_legend().set_visible(False)
        ax.set_title(sensor, fontsize=15 * relResolution[1], x=-0.05, y=0.35)
        if index == numSensors - 1:
            locs, _ = plt.xticks()
            locs = [locs[0], locs[-1]]
            locs = [second for second in range(0, numLines, frequency)]
            ax.set_xticks(locs, labels=[f"{str(tick / frequency)} s" for tick in locs])
        else:
            ax.set_xticks([])

    handles, _ = axs[0].get_legend_handles_labels()
    fig.set_title(recording.name or '')
    fig.legend(handles, legend_labels, loc='upper left')
    plt.show()
    save_path = "/Users/franz/Projects/dhc-lab/apps/ml-rainbow"
    fig.savefig(f'{save_path}/2D_Plot-Recording_xx{attr_suffix1}{attr_suffix2}{mag_suffix}.png')
    print(f"Saved to {save_path}/2D_Plot-Recording_xx{attr_suffix1}{attr_suffix2}{mag_suffix}.png")


def _columnNameMatches(column, column_prefixes) -> bool:
    for prefix in column_prefixes:
        if prefix in column:
            return True
    return False


def _calcMagnitude(row) -> float:
    return sqrt(sum([x ** 2 for x in row]))


if __name__ == '__main__':
    # plot_freeAccs_graph(0, columns=["Quat_W", "Quat_X"] useMagnitude=True, fixedValueRange=False)
    plot_freeAccs_graph(20, columns=['FreeAcc_Z'], useMagnitude=False, fixedValueRange=False, plotZeroLine=True,
                        relResolution=(1, 1))
