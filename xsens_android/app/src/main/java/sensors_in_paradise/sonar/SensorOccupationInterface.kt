package sensors_in_paradise.sonar

interface SensorOccupationInterface {
    /*Callback that should be called when sensor occupation
    status changed i.e.the sensors are currently in use by a recording or similar*/
    fun onSensorOccupationStatusChanged(occupied: Boolean)
}
