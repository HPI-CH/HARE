//  Copyright (c) 2003-2020 Xsens Technologies B.V. or subsidiaries worldwide.
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without modification,
//  are permitted provided that the following conditions are met:
//
//  1.      Redistributions of source code must retain the above copyright notice,
//           this list of conditions, and the following disclaimer.
//
//  2.      Redistributions in binary form must reproduce the above copyright notice,
//           this list of conditions, and the following disclaimer in the documentation
//           and/or other materials provided with the distribution.
//
//  3.      Neither the names of the copyright holders nor the names of their contributors
//           may be used to endorse or promote products derived from this software without
//           specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
//  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
//  THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
//  OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR
//  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.THE LAWS OF THE NETHERLANDS
//  SHALL BE EXCLUSIVELY APPLICABLE AND ANY DISPUTES SHALL BE FINALLY SETTLED UNDER THE RULES
//  OF ARBITRATION OF THE INTERNATIONAL CHAMBER OF COMMERCE IN THE HAGUE BY ONE OR MORE
//  ARBITRATORS APPOINTED IN ACCORDANCE WITH SAID RULES.
//

package com.xsens.dot.android.example.views;

import android.app.AlertDialog;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.le.ScanSettings;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.Observer;
import androidx.recyclerview.widget.DefaultItemAnimator;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.xsens.dot.android.example.R;
import com.xsens.dot.android.example.adapters.ScanAdapter;
import com.xsens.dot.android.example.databinding.FragmentScanBinding;
import com.xsens.dot.android.example.interfaces.BatteryChangedInterface;
import com.xsens.dot.android.example.interfaces.ScanClickInterface;
import com.xsens.dot.android.example.interfaces.SensorClickInterface;
import com.xsens.dot.android.example.viewmodels.BluetoothViewModel;
import com.xsens.dot.android.example.viewmodels.SensorViewModel;
import com.xsens.dot.android.sdk.interfaces.XsensDotScannerCallback;
import com.xsens.dot.android.sdk.models.XsensDotDevice;
import com.xsens.dot.android.sdk.utils.XsensDotScanner;

import java.util.ArrayList;
import java.util.HashMap;

import static com.xsens.dot.android.example.adapters.ScanAdapter.KEY_BATTERY_PERCENTAGE;
import static com.xsens.dot.android.example.adapters.ScanAdapter.KEY_BATTERY_STATE;
import static com.xsens.dot.android.example.adapters.ScanAdapter.KEY_CONNECTION_STATE;
import static com.xsens.dot.android.example.adapters.ScanAdapter.KEY_DEVICE;
import static com.xsens.dot.android.example.adapters.ScanAdapter.KEY_TAG;
import static com.xsens.dot.android.example.views.MainActivity.FRAGMENT_TAG_SCAN;
import static com.xsens.dot.android.sdk.models.XsensDotDevice.CONN_STATE_CONNECTED;
import static com.xsens.dot.android.sdk.models.XsensDotDevice.CONN_STATE_CONNECTING;
import static com.xsens.dot.android.sdk.models.XsensDotDevice.CONN_STATE_DISCONNECTED;
import static com.xsens.dot.android.sdk.models.XsensDotDevice.CONN_STATE_RECONNECTING;

/**
 * A fragment for scanned item.
 */
public class ScanFragment extends Fragment implements XsensDotScannerCallback, SensorClickInterface, ScanClickInterface, BatteryChangedInterface {

    private static final String TAG = ScanFragment.class.getSimpleName();

    // The view binder of ScanFragment
    private FragmentScanBinding mBinding;

    // The Bluetooth view model instance
    private BluetoothViewModel mBluetoothViewModel;

    // The devices view model instance
    private SensorViewModel mSensorViewModel;

    // The adapter for scanned device item
    private ScanAdapter mScanAdapter;

    // A list contains scanned Bluetooth device
    private ArrayList<HashMap<String, Object>> mScannedSensorList = new ArrayList<>();

    // The XsensDotScanner object
    private XsensDotScanner mXsDotScanner;

    // A variable for scanning flag
    private boolean mIsScanning = false;

    // A dialog during the connection
    private AlertDialog mConnectionDialog;

    /**
     * Get the instance of ScanFragment
     *
     * @return The instance of ScanFragment
     */
    public static ScanFragment newInstance() {

        return new ScanFragment();
    }

    @Override
    public void onAttach(@NonNull Context context) {

        super.onAttach(context);

        bindViewModel();
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {

        super.onCreateView(inflater, container, savedInstanceState);

        mBinding = FragmentScanBinding.inflate(LayoutInflater.from(getContext()));
        return mBinding.getRoot();
    }

    @Override
    public void onActivityCreated(@Nullable Bundle savedInstanceState) {

        super.onActivityCreated(savedInstanceState);

        mScanAdapter = new ScanAdapter(getContext(), mScannedSensorList);
        mScanAdapter.setSensorClickListener(this);
        mBinding.sensorRecyclerView.setLayoutManager(new LinearLayoutManager(getContext()));
        mBinding.sensorRecyclerView.setItemAnimator(new DefaultItemAnimator());
        mBinding.sensorRecyclerView.setAdapter(mScanAdapter);

        AlertDialog.Builder connectionDialogBuilder = new AlertDialog.Builder(getActivity());
        connectionDialogBuilder.setTitle(getString(R.string.connecting));
        connectionDialogBuilder.setMessage(getString(R.string.hint_connecting));
        mConnectionDialog = connectionDialogBuilder.create();

        // Set the SensorClickInterface instance to main activity.
        if (getActivity() != null) ((MainActivity) getActivity()).setScanTriggerListener(this);
    }

    @Override
    public void onResume() {

        super.onResume();

        // Notify main activity to refresh menu.
        MainActivity.sCurrentFragment = FRAGMENT_TAG_SCAN;
        if (getActivity() != null) getActivity().invalidateOptionsMenu();
    }

    @Override
    public void onDestroy() {

        super.onDestroy();

        // Stop scanning to let other apps to use scan function.
        if (mXsDotScanner != null) mXsDotScanner.stopScan();
        mBluetoothViewModel.updateScanState(false);
    }

    @Override
    public void onDetach() {

        super.onDetach();

        // Release all connections when app is destroyed.
        mSensorViewModel.disconnectAllSensors();
    }

    @Override
    public void onScanTriggered(boolean triggered) {

        if (triggered) {
            // Disconnect to all sensors to make sure the connection has been released.
            mSensorViewModel.disconnectAllSensors();
            // This line is for connecting and reconnecting device.
            // Because they don't triggered onXsensDotConnectionChanged() function to remove sensor from list.
            mSensorViewModel.removeAllDevice();

            mScannedSensorList.clear();
            mScanAdapter.notifyDataSetChanged();
            mIsScanning = mXsDotScanner.startScan();

        } else {
            // If success for stopping, it will return True from SDK. So use !(not) here.
            mIsScanning = !mXsDotScanner.stopScan();
        }

        mBluetoothViewModel.updateScanState(mIsScanning);
    }

    @Override
    public void onXsensDotScanned(BluetoothDevice device, int rssi) {

        Log.i(TAG, "onXsensDotScanned() - Name: " + device.getName() + ", Address: " + device.getAddress());

        if (isAdded()) {

            // Use the mac address as UID to filter the same scan result.
            boolean isExist = false;
            for (HashMap<String, Object> map : mScannedSensorList) {

                if (((BluetoothDevice) map.get(KEY_DEVICE)).getAddress().equals(device.getAddress()))
                    isExist = true;
            }

            if (!isExist) {

                // The original connection state is Disconnected.
                // Also set tag, battery state, battery percentage to default value.
                HashMap<String, Object> map = new HashMap<>();
                map.put(KEY_DEVICE, device);
                map.put(KEY_CONNECTION_STATE, CONN_STATE_DISCONNECTED);
                map.put(KEY_TAG, "");
                map.put(KEY_BATTERY_STATE, -1);
                map.put(KEY_BATTERY_PERCENTAGE, -1);
                mScannedSensorList.add(map);

                mScanAdapter.notifyItemInserted(mScannedSensorList.size() - 1);
            }
        }
    }

    @Override
    public void onSensorClick(View v, int position) {

        // If success for stopping, it will return True from SDK. So use !(not) here.
        mIsScanning = !mXsDotScanner.stopScan();
        // Notify main activity to update the scan button.
        mBluetoothViewModel.updateScanState(false);

        int state = mScanAdapter.getConnectionState(position);
        BluetoothDevice device = mScanAdapter.getDevice(position);

        /**
         * state = 0 : Disconnected
         * state = 1 : Connecting
         * state = 2 : Connected
         * state = 4 : Reconnecting
         */
        switch (state) {

            case CONN_STATE_DISCONNECTED:

                mConnectionDialog.show();
                // The sensor isn't exist in the mSensorList(SensorViewModel), try to connect and add it.
                mSensorViewModel.connectSensor(getContext(), device);
                break;

            case CONN_STATE_CONNECTING:

                mScanAdapter.updateConnectionState(position, CONN_STATE_DISCONNECTED);
                mScanAdapter.notifyItemChanged(position);
                // This line is necessary to close Bluetooth gatt.
                mSensorViewModel.disconnectSensor(device.getAddress());
                // Remove this sensor from device list.
                mSensorViewModel.removeDevice(device.getAddress());
                break;

            case CONN_STATE_CONNECTED:

                mSensorViewModel.disconnectSensor(device.getAddress());
                // No need to call removeDevice() function, just wait for onXsensDotConnectionChanged() callback function.
                break;

            case CONN_STATE_RECONNECTING:

                mScanAdapter.updateConnectionState(position, CONN_STATE_DISCONNECTED);
                mScanAdapter.notifyItemChanged(position);
                // This line is necessary to close Bluetooth gatt.
                mSensorViewModel.cancelReconnection(device.getAddress());
                // Remove this sensor from device list.
                mSensorViewModel.removeDevice(device.getAddress());
                break;
        }
    }

    /**
     * Initialize and observe view models.
     */
    private void bindViewModel() {

        if (getActivity() != null) {

            mBluetoothViewModel = BluetoothViewModel.getInstance(getActivity());
            mSensorViewModel = SensorViewModel.getInstance(getActivity());

            mBluetoothViewModel.isBluetoothEnabled().observe(this, new Observer<Boolean>() {

                @Override
                public void onChanged(Boolean enabled) {

                    Log.d(TAG, "isBluetoothEnabled = " + enabled);

                    if (enabled) {
                        initXsDotScanner();
                    } else {
                        mIsScanning = false;
                        mBluetoothViewModel.updateScanState(false);
                    }
                }
            });

            mSensorViewModel.getConnectionChangedDevice().observe(this, new Observer<XsensDotDevice>() {

                @Override
                public void onChanged(XsensDotDevice device) {

                    String address = device.getAddress();
                    int state = device.getConnectionState();
                    Log.d(TAG, "getConnectionChangedDevice() - address = " + address + ", state = " + state);

                    for (HashMap<String, Object> map : mScannedSensorList) {

                        BluetoothDevice _device = (BluetoothDevice) map.get(KEY_DEVICE);

                        if (_device != null) {

                            String _address = _device.getAddress();
                            // Update connection state by the same mac address.
                            if (_address.equals(address)) {

                                map.put(KEY_CONNECTION_STATE, state);
                                mScanAdapter.notifyDataSetChanged();
                            }
                        }
                    }

                    switch (state) {

                        case CONN_STATE_CONNECTED:

                            if (mConnectionDialog.isShowing()) mConnectionDialog.dismiss();
                            break;
                    }
                }
            });

            mSensorViewModel.getTagChangedDevice().observe(this, new Observer<XsensDotDevice>() {

                @Override
                public void onChanged(XsensDotDevice device) {

                    String address = device.getAddress();
                    String tag = device.getTag();

                    mScanAdapter.updateTag(address, tag);
                    mScanAdapter.notifyDataSetChanged();
                    Log.d(TAG, "getTagChangedDevice() - address = " + address + ", tag = " + tag);
                }
            });

            mSensorViewModel.setBatteryChangedCallback(this);
        }
    }

    /**
     * Setup for Xsens DOT scanner.
     */
    private void initXsDotScanner() {

        if (mXsDotScanner == null) {

            mXsDotScanner = new XsensDotScanner(getContext(), this);
            mXsDotScanner.setScanMode(ScanSettings.SCAN_MODE_BALANCED);
        }
    }

    @Override
    public void onBatteryChanged(String address, int state, int percentage) {

        Log.d(TAG, "onBatteryChanged() - address = " + address + ", state = " + state + ", percentage = " + percentage);

        mScanAdapter.updateBattery(address, state, percentage);

        if (getActivity() != null) {
            // This event is coming from background thread, use UI thread to update item.
            getActivity().runOnUiThread(new Runnable() {
                @Override
                public void run() {

                    mScanAdapter.notifyDataSetChanged();
                }
            });
        }
    }
}
