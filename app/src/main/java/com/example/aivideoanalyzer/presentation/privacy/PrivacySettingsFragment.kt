package com.example.aivideoanalyzer.presentation.privacy

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import com.example.aivideoanalyzer.databinding.FragmentPrivacySettingsBinding
import com.example.aivideoanalyzer.presentation.ui.ViewModelFactory
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import java.text.SimpleDateFormat
import java.util.*

class PrivacySettingsFragment : Fragment() {

    private var _binding: FragmentPrivacySettingsBinding? = null
    private val binding get() = _binding!!

    private val viewModel: PrivacyViewModel by viewModels {
        ViewModelFactory(requireActivity().application)
    }

    private val dateFormat = SimpleDateFormat("MMM dd, yyyy 'at' HH:mm", Locale.getDefault())

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentPrivacySettingsBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        setupUI()
        observeViewModel()
    }

    private fun setupUI() {
        // Privacy overview section
        binding.privacyOverviewCard.setOnClickListener {
            showPrivacyOverview()
        }

        // Data usage section
        binding.dataUsageCard.setOnClickListener {
            showDataUsageDetails()
        }

        // Consent management
        binding.revokeConsentButton.setOnClickListener {
            showRevokeConsentDialog()
        }

        // Export data
        binding.exportDataButton.setOnClickListener {
            exportPrivacyData()
        }

        // Clear all data
        binding.clearDataButton.setOnClickListener {
            showClearDataDialog()
        }

        // View privacy policy
        binding.viewPrivacyPolicyButton.setOnClickListener {
            showFullPrivacyPolicy()
        }
    }

    private fun observeViewModel() {
        viewModel.privacySettings.observe(viewLifecycleOwner) { settings ->
            updateUI(settings)
        }
    }

    private fun updateUI(settings: PrivacyViewModel.PrivacySettings) {
        // Update consent status
        binding.videoConsentStatus.text = if (settings.hasVideoProcessingConsent) {
            "✅ Granted"
        } else {
            "❌ Not Granted"
        }

        binding.privacyPolicyStatus.text = if (settings.hasPrivacyPolicyAccepted) {
            "✅ Accepted"
        } else {
            "❌ Not Accepted"
        }

        // Update consent date
        if (settings.consentTimestamp > 0) {
            binding.consentDate.text = "Consent given on: ${dateFormat.format(Date(settings.consentTimestamp))}"
            binding.consentDate.visibility = View.VISIBLE
        } else {
            binding.consentDate.visibility = View.GONE
        }

        // Update button states
        binding.revokeConsentButton.isEnabled = settings.hasVideoProcessingConsent

        // Update data usage summary
        val dataUsage = viewModel.getDataUsageSummary()
        updateDataUsageSummary(dataUsage)
    }

    private fun updateDataUsageSummary(dataUsage: PrivacyViewModel.DataUsageSummary) {
        binding.localProcessingStatus.text = if (dataUsage.videosProcessedLocally) "✅ Local Only" else "❌ Remote"
        binding.thirdPartyStatus.text = if (!dataUsage.dataSharedWithThirdParties) "✅ Not Shared" else "❌ Shared"
        binding.cloudStorageStatus.text = if (!dataUsage.dataStoredInCloud) "✅ Device Only" else "❌ Cloud Storage"
        binding.trainingDataStatus.text = if (!dataUsage.dataUsedForTraining) "✅ Not Used" else "❌ Used for Training"
    }

    private fun showPrivacyOverview() {
        val overview = """
            🔒 Your Privacy at a Glance
            
            ✅ Videos processed locally on your device
            ✅ No data sent to remote servers
            ✅ No cloud storage of your videos
            ✅ No use of your data for AI training
            ✅ Temporary files automatically deleted
            ✅ You control all your data
            
            🚫 We Never:
            • Upload your videos anywhere
            • Store videos in the cloud
            • Share data with third parties
            • Use your videos to improve our models
            • Process videos on remote servers
            
            ✅ We Only:
            • Process videos locally on your device
            • Store analysis results on your device
            • Allow you to export results if you choose
            • Collect anonymous app usage statistics
        """.trimIndent()

        MaterialAlertDialogBuilder(requireContext())
            .setTitle("Privacy Overview")
            .setMessage(overview)
            .setPositiveButton("OK", null)
            .show()
    }

    private fun showDataUsageDetails() {
        val details = """
            📊 Detailed Data Usage Information
            
            🎥 Video Processing:
            • Videos are processed entirely on your device
            • No video data ever leaves your phone
            • Temporary files are automatically deleted after processing
            
            💾 Data Storage:
            • Analysis results stored locally in app's private folder
            • You can delete results anytime
            • Exported files go to your Documents folder
            
            📡 Network Usage:
            • App can work completely offline
            • Internet only used for app updates (optional)
            • No video or analysis data transmitted
            
            🔄 Data Sharing:
            • Zero data shared with third parties
            • No analytics or tracking of your videos
            • Only anonymous app usage statistics collected
            
            🗑️ Data Deletion:
            • Videos: Deleted immediately after processing
            • Temporary files: Auto-deleted after each session
            • Results: Kept until you manually delete them
        """.trimIndent()

        MaterialAlertDialogBuilder(requireContext())
            .setTitle("Data Usage Details")
            .setMessage(details)
            .setPositiveButton("OK", null)
            .show()
    }

    private fun showRevokeConsentDialog() {
        MaterialAlertDialogBuilder(requireContext())
            .setTitle("Revoke Consent")
            .setMessage("""
                Are you sure you want to revoke your consent for video processing?
                
                This will:
                • Prevent the app from processing new videos
                • Keep existing analysis results (unless you delete them)
                • Require you to grant consent again to use the app
                
                You can re-grant consent at any time.
            """.trimIndent())
            .setPositiveButton("Revoke") { _, _ ->
                viewModel.revokeVideoProcessingConsent()
                showMessage("Consent revoked. You'll need to grant consent again to process videos.")
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun exportPrivacyData() {
        val privacyData = viewModel.exportPrivacyData()

        val exportText = """
            AI Video Analyzer - Privacy Data Export
            Generated on: ${dateFormat.format(Date())}
            
            CONSENT STATUS:
            Video Processing Consent: ${if (privacyData.videoProcessingConsent) "Granted" else "Not Granted"}
            Privacy Policy Accepted: ${if (privacyData.privacyPolicyAccepted) "Yes" else "No"}
            Consent Date: ${if (privacyData.consentTimestamp > 0) dateFormat.format(Date(privacyData.consentTimestamp)) else "N/A"}
            Privacy Policy Version: ${privacyData.privacyPolicyVersion}
            
            DATA PRACTICES:
            Videos Processed Locally: Yes
            Data Shared with Third Parties: No
            Data Stored in Cloud: No
            Data Used for Training: No
            Temporary Files Auto-Deleted: Yes
            
            TECHNICAL DETAILS:
            Local Processing: PyTorch Mobile models run on device
            Storage Location: Device private storage only
            Network Usage: No video data transmitted
            Data Retention: User-controlled deletion
        """.trimIndent()

        // Create share intent
        val shareIntent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_TEXT, exportText)
            putExtra(Intent.EXTRA_SUBJECT, "AI Video Analyzer - Privacy Data Export")
        }

        startActivity(Intent.createChooser(shareIntent, "Export Privacy Data"))
    }

    private fun showClearDataDialog() {
        MaterialAlertDialogBuilder(requireContext())
            .setTitle("Clear All Data")
            .setMessage("""
                This will permanently delete:
                • All privacy consent records
                • All video analysis results
                • All app preferences
                
                You will need to grant consent again to use the app.
                
                This action cannot be undone.
            """.trimIndent())
            .setPositiveButton("Clear All Data") { _, _ ->
                viewModel.clearAllPrivacyData()
                showMessage("All data cleared. Please restart the app.")
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showFullPrivacyPolicy() {
        PrivacyConsentDialogFragment.newInstance().show(
            parentFragmentManager,
            PrivacyConsentDialogFragment.TAG
        )
    }

    private fun showMessage(message: String) {
        MaterialAlertDialogBuilder(requireContext())
            .setMessage(message)
            .setPositiveButton("OK", null)
            .show()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}