package com.example.aivideoanalyzer.presentation.privacy

import android.app.Dialog
import android.os.Bundle
import android.text.method.LinkMovementMethod
import androidx.core.text.HtmlCompat
import androidx.fragment.app.DialogFragment
import androidx.fragment.app.activityViewModels
import com.example.aivideoanalyzer.databinding.DialogPrivacyConsentBinding
import com.google.android.material.dialog.MaterialAlertDialogBuilder

class PrivacyConsentDialogFragment : DialogFragment() {

    private var _binding: DialogPrivacyConsentBinding? = null
    private val binding get() = _binding!!

    private val viewModel: PrivacyViewModel by activityViewModels()

    companion object {
        const val TAG = "PrivacyConsentDialog"

        fun newInstance(): PrivacyConsentDialogFragment {
            return PrivacyConsentDialogFragment()
        }
    }

    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        _binding = DialogPrivacyConsentBinding.inflate(layoutInflater)

        setupUI()
        observeViewModel()

        return MaterialAlertDialogBuilder(requireContext())
            .setView(binding.root)
            .setCancelable(false)
            .create()
    }

    private fun setupUI() {
        // Set up privacy policy text with HTML formatting
        val privacyText = """
            <h3>Privacy Policy & Video Processing Consent</h3>
            
            <h4>🔒 Your Privacy is Our Priority</h4>
            <p>We are committed to protecting your privacy and ensuring transparent data handling.</p>
            
            <h4>📱 Local Processing Only</h4>
            <p><strong>Your videos NEVER leave your device.</strong> All AI analysis is performed locally on your phone using on-device machine learning models.</p>
            
            <h4>🚫 What We DON'T Do</h4>
            <ul>
            <li>❌ Upload your videos to remote servers</li>
            <li>❌ Store your videos in the cloud</li>
            <li>❌ Use your videos for AI model training</li>
            <li>❌ Share your videos with third parties</li>
            <li>❌ Analyze your videos on external servers</li>
            </ul>
            
            <h4>✅ What We DO</h4>
            <ul>
            <li>✅ Process videos locally on your device</li>
            <li>✅ Store analysis results only on your device</li>
            <li>✅ Allow you to export/share results at your discretion</li>
            <li>✅ Delete temporary files after processing</li>
            </ul>
            
            <h4>📊 Data Collection</h4>
            <p>We collect minimal, non-personal data:</p>
            <ul>
            <li>App usage statistics (anonymous)</li>
            <li>Crash reports (no personal data)</li>
            <li>Performance metrics (device type, processing times)</li>
            </ul>
            
            <h4>🗑️ Data Retention</h4>
            <ul>
            <li>Videos: Deleted immediately after processing</li>
            <li>Analysis results: Stored locally until you delete them</li>
            <li>Temporary files: Auto-deleted after each session</li>
            </ul>
            
            <h4>🔧 Your Rights</h4>
            <ul>
            <li>View all stored data on your device</li>
            <li>Delete any data at any time</li>
            <li>Revoke consent and stop using the app</li>
            <li>Export your analysis results</li>
            </ul>
            
            <p><small>By continuing, you consent to local video processing and agree to our privacy practices. You can revoke this consent at any time in the app settings.</small></p>
        """.trimIndent()

        binding.privacyPolicyText.text = HtmlCompat.fromHtml(privacyText, HtmlCompat.FROM_HTML_MODE_LEGACY)
        binding.privacyPolicyText.movementMethod = LinkMovementMethod.getInstance()

        // Set up consent checkboxes
        binding.videoProcessingConsent.text = "I understand that my videos will be processed locally on this device and will not be uploaded to any servers"
        binding.privacyPolicyConsent.text = "I agree to the privacy policy and data handling practices described above"

        // Set up buttons
        binding.acceptButton.setOnClickListener {
            if (validateConsent()) {
                grantConsent()
            }
        }

        binding.declineButton.setOnClickListener {
            declineConsent()
        }

        binding.moreInfoButton.setOnClickListener {
            showDetailedPrivacyInfo()
        }

        // Initially disable accept button
        updateAcceptButtonState()

        // Listen for checkbox changes
        binding.videoProcessingConsent.setOnCheckedChangeListener { _, _ -> updateAcceptButtonState() }
        binding.privacyPolicyConsent.setOnCheckedChangeListener { _, _ -> updateAcceptButtonState() }
    }

    private fun observeViewModel() {
        viewModel.consentResult.observe(this) { success ->
            if (success) {
                dismiss()
            }
        }
    }

    private fun validateConsent(): Boolean {
        val videoConsent = binding.videoProcessingConsent.isChecked
        val privacyConsent = binding.privacyPolicyConsent.isChecked

        if (!videoConsent) {
            binding.videoProcessingConsent.error = "This consent is required to use the app"
            return false
        }

        if (!privacyConsent) {
            binding.privacyPolicyConsent.error = "Please agree to the privacy policy"
            return false
        }

        return true
    }

    private fun grantConsent() {
        viewModel.grantConsent()
    }

    private fun declineConsent() {
        MaterialAlertDialogBuilder(requireContext())
            .setTitle("Exit App")
            .setMessage("Without your consent, the app cannot function. The app will now close.")
            .setPositiveButton("Exit") { _, _ ->
                requireActivity().finish()
            }
            .setNegativeButton("Review Again") { _, _ ->
                // Stay on dialog
            }
            .show()
    }

    private fun showDetailedPrivacyInfo() {
        val detailedInfo = """
            <h3>Detailed Technical Information</h3>
            
            <h4>🔍 How Local Processing Works</h4>
            <p>Our app uses PyTorch Mobile models that run entirely on your device. When you select a video:</p>
            <ol>
            <li>Video is temporarily copied to app's private storage</li>
            <li>Frames are extracted and processed locally</li>
            <li>AI analysis happens on your device's CPU/GPU</li>
            <li>Results are saved to your device only</li>
            <li>Original video and temporary files are deleted</li>
            </ol>
            
            <h4>📂 File Storage Locations</h4>
            <ul>
            <li><strong>Temporary files:</strong> /Android/data/[app]/cache/ (auto-deleted)</li>
            <li><strong>Results:</strong> /Android/data/[app]/files/ (user controlled)</li>
            <li><strong>Exports:</strong> /Documents/AIVideoAnalyzer/ (user accessible)</li>
            </ul>
            
            <h4>🔒 Security Measures</h4>
            <ul>
            <li>No network permissions for video data</li>
            <li>Encrypted local storage</li>
            <li>Automatic cleanup of temporary files</li>
            <li>No background processing</li>
            </ul>
            
            <h4>📱 Permissions Used</h4>
            <ul>
            <li><strong>Storage:</strong> To read/write videos and results</li>
            <li><strong>Camera (optional):</strong> To record videos directly</li>
            </ul>
            
            <h4>🌐 No Internet Required</h4>
            <p>The app works completely offline. Internet is only used for:</p>
            <ul>
            <li>Checking for app updates (optional)</li>
            <li>Anonymous crash reporting (if enabled)</li>
            </ul>
        """.trimIndent()

        MaterialAlertDialogBuilder(requireContext())
            .setTitle("Technical Details")
            .setMessage(HtmlCompat.fromHtml(detailedInfo, HtmlCompat.FROM_HTML_MODE_LEGACY))
            .setPositiveButton("OK", null)
            .show()
    }

    private fun updateAcceptButtonState() {
        val bothChecked = binding.videoProcessingConsent.isChecked &&
                binding.privacyPolicyConsent.isChecked
        binding.acceptButton.isEnabled = bothChecked
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}