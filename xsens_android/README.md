[![linter](https://github.com/Sensors-in-Paradise/android-native-xsens/actions/workflows/linter.yml/badge.svg)](https://github.com/Sensors-in-Paradise/android-native-xsens/actions/workflows/linter.yml)
[![unit-test](https://github.com/Sensors-in-Paradise/android-native-xsens/actions/workflows/unit-test.yml/badge.svg)](https://github.com/Sensors-in-Paradise/android-native-xsens/actions/workflows/unit-test.yml)
[![instrumented-test](https://github.com/Sensors-in-Paradise/android-native-xsens/actions/workflows/instrumented-test.yml/badge.svg)](https://github.com/Sensors-in-Paradise/android-native-xsens/actions/workflows/instrumented-test.yml)

# HARE - A Generic HAR App

This is a tool for recording data through Xsens DOT sensors and labeling it when saving on the Device. It is designed to be very generic so that changing the sensors is possible while keeping most of the existing code and functionality. 
![image](https://user-images.githubusercontent.com/29177177/185871656-5681deb4-c0e4-4380-925e-a94605fd8aa6.png)

## Available functionalities

- Connection and synchronisation for Xsens DOT sensors
- Labeling with one or multiple labels per recording
- On-Device prediction after adding a model
- Ability to use multiple use cases on one device
- Chance to display current sensor rotation and received data
- Additional camera recording
- Pose estimation from camera to store anonymous visual data
- Relabeling during and after recording
- Optimal Sensor Placement Estimation
- Simple Connection to any WebDAV Cloud to make storing data easy
- Data visualisation on device
- On-Device training

## How to use the app

The Connection Screen works well with Xsens DOT Sensors. For connected Sensors you can make them flash their LED to identify them making a long press on their item. After their first connection they will be remembered with their name and set.

The Recording Screen is used for making, labeling and storing recordings and to perform sensor placement estimations. You can add and select subjects as well as the performed action to record. There can be only one subject per recording and the one currently selected at the end of the recording will be saved. For Activities however there can be multiple ones. When you open the Dialog a timestamp will be saved instantly with the label you select afterwards. This way you can take your time to select the new label and the timestamp will not change. If you want to discard this timestamp (because the activity did not change) you can close the dialog. <br />
If you did not select a Label and the app crashes/ is stopped by android then the null - activity label and unknown subject will be chosen as a default value.
During an recording you can use the activities tab to change labels and see the label history of this recording. For Changing Timestamps together with Labels you should edit the recording afterwards using the video recording for instance. Please note that we do not record audio together with video data. <br />
To estimate best placements for a certain number of sensors, you can long press on recordings with pose estimation file. The app automatically switches to selection mode, where you can further add recordings on which the estimation should be performed. The follwoing body parts are taken into consideration: Head, Right Shoulder (RS), Left Shoulder (LS), Right Elbow (RE), Left Elbow (LE), Right Wrist (RW), Left Wrist (LW), Hip, Right Knee (RK), Left Knee (LK), Right Ankle (RA), Left Ankle (LA).

On the Data Screen "Filter for Training" hides the amount of collected data that has already been used for training. Only the recordings in the currently selected subdirectory of a use case will be considered on this screen.

On the Prediction Screen you can see the confidence of the model for the highest predictions at the top. Below a history will be visible for the predicted activities. If one prediction stays the same then the space for it widens and the confidence for that label will be displayed as the average.

Not all Functions are activated by default and can be added in the settings menu at the top right.

## Architecture overview
In the following a quick summary of the app architecture can be found. It is not to be seen as a complete guide but rather as a starting point for continued development of the app.
### Screens
The app consists of one single activity called `MainActivity`(except the Settings Activity) which consists of 4 `screens` which can be switched between using a tab layout.
For each screen there is a package with the prefix `screen` (e.g. `screen_connection`) that contains the corresponding screen class with `Screen` suffix (e.g. `ConnectionScreen`). These screen classes implement the `ScreenInterface` which passes the lifecycle events of the MainActivity as well as other events to this class.

The `ConnectionScreen` class handles the UI and functionality for managing connections with the XSensDot devices. It also implements `XsensDotScannerCallback` and` XsensDotDeviceCallback` from the XSens Dot SDK to interface the XSens Dot devices. Since we don't need to listen for all of the events from `XsensDotScannerCallback` and` XsensDotDeviceCallback` in the other screens, we have simplified the callbacks with the `ConnectionInterface`. It can be implemented by the other screens to receive data from the sensors or be notified when devices connect or disconnect. Then the instance of the class implementing the interface must be added to the `ConnectionScreen` instance via `connectionScreen.addConnectionInterface(connectionInterface)`. This is typically done in the `MainActivity` since one has access to all screen instances there.

## External sources used

[Xsens DOT Manual](https://www.xsens.com/hubfs/Downloads/Manuals/Xsens%20DOT%20User%20Manual.pdf)

[WebDAV Client on Github](https://github.com/thegrizzlylabs/sardine-android)

[Charts Library on Github](https://github.com/PhilJay/MPAndroidChart)

[Tree View Library on Github](https://github.com/bmelnychuk/AndroidTreeView)

