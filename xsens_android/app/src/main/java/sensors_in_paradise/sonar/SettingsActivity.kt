package sensors_in_paradise.sonar

import android.app.Activity
import android.content.Intent
import android.graphics.drawable.ColorDrawable
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.preference.Preference
import androidx.preference.PreferenceFragmentCompat
import androidx.preference.SwitchPreference
import sensors_in_paradise.sonar.util.PreferencesHelper

class SettingsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.settings_activity)
        if (savedInstanceState == null) {
            supportFragmentManager
                .beginTransaction()
                .replace(R.id.settings, SettingsFragment())
                .commit()
        }
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.setBackgroundDrawable(ColorDrawable(getColor(R.color.colorPrimary)))
    }

    class SettingsFragment : PreferenceFragmentCompat() {
        private var editorStickmanBackgroundPref: SwitchPreference? = null

        override fun onCreatePreferences(savedInstanceState: Bundle?, rootKey: String?) {
            setPreferencesFromResource(R.xml.root_preferences, rootKey)

            editorStickmanBackgroundPref = findPreference("showPoseBackground") as SwitchPreference?
            editorStickmanBackgroundPref?.onPreferenceChangeListener =
                Preference.OnPreferenceChangeListener { _, newValue ->
                    if (newValue as Boolean) {
                        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
                        intent.addCategory(Intent.CATEGORY_OPENABLE)
                        intent.type = "image/png"
                        resultLauncher.launch(intent)
                    }
                    true
                }
        }

        private var resultLauncher =
            registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
                val editor =
                    this.context?.let { PreferencesHelper.getSharedPreferences(it).edit() }
                if (editor != null) {
                    if (result.resultCode == Activity.RESULT_OK) {
                        val data = result.data
                        val imageUri = data?.data
                        val takeFlags = (data?.flags?.and(
                            (Intent.FLAG_GRANT_READ_URI_PERMISSION
                                    or Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
                        ))
                        try {
                            requireActivity().contentResolver.takePersistableUriPermission(
                                imageUri!!,
                                takeFlags!!
                            )
                            editor.putString("poseSequenceBackground", imageUri.toString())
                        } catch (e: SecurityException) {
                            e.printStackTrace()
                            editor.putBoolean("showPoseBackground", false)
                            editorStickmanBackgroundPref?.isChecked = false
                        }
                    } else {
                        editor.putBoolean("showPoseBackground", false)
                        editorStickmanBackgroundPref?.isChecked = false
                    }
                    editor.apply()
                }
            }
    }
}
