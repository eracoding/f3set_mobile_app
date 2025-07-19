package com.example.aivideoanalyzer.presentation.ui.main

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.navigation.fragment.NavHostFragment
import androidx.navigation.ui.setupWithNavController
import com.example.aivideoanalyzer.R
import com.example.aivideoanalyzer.databinding.ActivityMainBinding
import com.example.aivideoanalyzer.presentation.privacy.PrivacyConsentDialogFragment
import com.example.aivideoanalyzer.presentation.privacy.PrivacyViewModel
import com.example.aivideoanalyzer.presentation.privacy.PrivacyManager
import com.example.aivideoanalyzer.presentation.ui.ViewModelFactory
import com.google.android.material.snackbar.Snackbar

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val viewModel: MainViewModel by viewModels { ViewModelFactory(application) }
    private val privacyViewModel: PrivacyViewModel by viewModels { ViewModelFactory(application) }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.values.all { it }
        if (!allGranted) {
            showPermissionDeniedMessage()
        } else {
            // Permissions granted, check privacy consent
            checkPrivacyConsent()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupNavigation()
        observeViewModels()

        // Check permissions and privacy on app start
        checkPermissionsAndPrivacy()
    }

    private fun setupNavigation() {
        val navHostFragment = supportFragmentManager
            .findFragmentById(R.id.nav_host_fragment) as NavHostFragment
        val navController = navHostFragment.navController

        binding.bottomNavigation.setupWithNavController(navController)

        // Handle reselection to scroll to top
        binding.bottomNavigation.setOnItemReselectedListener { menuItem ->
            // Handle reselection if needed
        }
    }

    private fun observeViewModels() {
        viewModel.appState.observe(this) { state ->
            when (state) {
                is MainViewModel.AppState.Loading -> showLoading()
                is MainViewModel.AppState.Ready -> hideLoading()
                is MainViewModel.AppState.Error -> showError(state.message)
            }
        }

        privacyViewModel.privacyConsentStatus.observe(this) { status ->
            when (status) {
                PrivacyManager.PrivacyConsentStatus.PRIVACY_POLICY_REQUIRED,
                PrivacyManager.PrivacyConsentStatus.VIDEO_CONSENT_REQUIRED -> {
                    showPrivacyConsentDialog()
                }
                PrivacyManager.PrivacyConsentStatus.ALL_GRANTED -> {
                    // All good, app can function normally
                }
            }
        }
    }

    private fun checkPermissionsAndPrivacy() {
        if (hasRequiredPermissions()) {
            checkPrivacyConsent()
        } else {
            requestPermissions()
        }
    }

    private fun hasRequiredPermissions(): Boolean {
        val permissions = getRequiredPermissions()
        return permissions.all { permission ->
            ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun getRequiredPermissions(): Array<String> {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            arrayOf(
                Manifest.permission.READ_MEDIA_VIDEO,
                Manifest.permission.READ_MEDIA_IMAGES
            )
        } else {
            arrayOf(
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
            )
        }
    }

    private fun requestPermissions() {
        val permissions = getRequiredPermissions()

        // Show rationale if needed
        val shouldShowRationale = permissions.any { permission ->
            shouldShowRequestPermissionRationale(permission)
        }

        if (shouldShowRationale) {
            showPermissionRationale {
                permissionLauncher.launch(permissions)
            }
        } else {
            permissionLauncher.launch(permissions)
        }
    }

    private fun showPermissionRationale(onAccept: () -> Unit) {
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Storage Permission Required")
            .setMessage("""
                AI Video Analyzer needs storage permission to:
                
                📱 Access videos from your device
                💾 Save analysis results
                📤 Export results to your Documents folder
                
                🔒 Privacy Note:
                Your videos are processed locally on your device and never uploaded to any servers.
            """.trimIndent())
            .setPositiveButton("Grant Permission") { _, _ ->
                onAccept()
            }
            .setNegativeButton("Exit App") { _, _ ->
                finish()
            }
            .setCancelable(false)
            .show()
    }

    private fun checkPrivacyConsent() {
        if (privacyViewModel.needsPrivacyConsent()) {
            showPrivacyConsentDialog()
        }
    }

    private fun showPrivacyConsentDialog() {
        val existingFragment = supportFragmentManager.findFragmentByTag(PrivacyConsentDialogFragment.TAG)
        if (existingFragment == null) {
            PrivacyConsentDialogFragment.newInstance().show(
                supportFragmentManager,
                PrivacyConsentDialogFragment.TAG
            )
        }
    }

    private fun showPermissionDeniedMessage() {
        Snackbar.make(
            binding.root,
            "Storage permission is required to process videos. The app cannot function without it.",
            Snackbar.LENGTH_INDEFINITE
        ).setAction("Grant") {
            requestPermissions()
        }.setActionTextColor(getColor(R.color.md_theme_primary))
            .show()
    }

    private fun showLoading() {
        // Show loading indicator if needed
    }

    private fun hideLoading() {
        // Hide loading indicator
    }

    private fun showError(message: String) {
        Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG).show()
    }
}