package com.example.aivideoanalyzer.presentation.privacy

import android.content.Context
import android.content.SharedPreferences
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData

/**
 * Manages user privacy preferences and consent
 */
class PrivacyManager(private val context: Context) {

    companion object {
        private const val PREFS_NAME = "privacy_preferences"
        private const val KEY_VIDEO_PROCESSING_CONSENT = "video_processing_consent"
        private const val KEY_PRIVACY_POLICY_ACCEPTED = "privacy_policy_accepted"
        private const val KEY_FIRST_LAUNCH = "first_launch"
        private const val KEY_CONSENT_TIMESTAMP = "consent_timestamp"
        private const val KEY_PRIVACY_POLICY_VERSION = "privacy_policy_version"

        // Current privacy policy version - increment when policy changes
        private const val CURRENT_PRIVACY_POLICY_VERSION = 1
    }

    private val sharedPrefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private val _privacyConsentStatus = MutableLiveData<PrivacyConsentStatus>()
    val privacyConsentStatus: LiveData<PrivacyConsentStatus> = _privacyConsentStatus

    init {
        updateConsentStatus()
    }

    /**
     * Check if this is the first app launch
     */
    fun isFirstLaunch(): Boolean {
        return sharedPrefs.getBoolean(KEY_FIRST_LAUNCH, true)
    }

    /**
     * Mark first launch as completed
     */
    fun markFirstLaunchCompleted() {
        sharedPrefs.edit()
            .putBoolean(KEY_FIRST_LAUNCH, false)
            .apply()
    }

    /**
     * Check if user has given consent for video processing
     */
    fun hasVideoProcessingConsent(): Boolean {
        return sharedPrefs.getBoolean(KEY_VIDEO_PROCESSING_CONSENT, false)
    }

    /**
     * Check if privacy policy has been accepted
     */
    fun hasPrivacyPolicyAccepted(): Boolean {
        val accepted = sharedPrefs.getBoolean(KEY_PRIVACY_POLICY_ACCEPTED, false)
        val version = sharedPrefs.getInt(KEY_PRIVACY_POLICY_VERSION, 0)
        return accepted && version >= CURRENT_PRIVACY_POLICY_VERSION
    }

    /**
     * Grant video processing consent
     */
    fun grantVideoProcessingConsent() {
        sharedPrefs.edit()
            .putBoolean(KEY_VIDEO_PROCESSING_CONSENT, true)
            .putLong(KEY_CONSENT_TIMESTAMP, System.currentTimeMillis())
            .apply()
        updateConsentStatus()
    }

    /**
     * Revoke video processing consent
     */
    fun revokeVideoProcessingConsent() {
        sharedPrefs.edit()
            .putBoolean(KEY_VIDEO_PROCESSING_CONSENT, false)
            .apply()
        updateConsentStatus()
    }

    /**
     * Accept privacy policy
     */
    fun acceptPrivacyPolicy() {
        sharedPrefs.edit()
            .putBoolean(KEY_PRIVACY_POLICY_ACCEPTED, true)
            .putInt(KEY_PRIVACY_POLICY_VERSION, CURRENT_PRIVACY_POLICY_VERSION)
            .putLong(KEY_CONSENT_TIMESTAMP, System.currentTimeMillis())
            .apply()
        updateConsentStatus()
    }

    /**
     * Get consent timestamp
     */
    fun getConsentTimestamp(): Long {
        return sharedPrefs.getLong(KEY_CONSENT_TIMESTAMP, 0)
    }

    /**
     * Check if user needs to see privacy consent dialog
     */
    fun needsPrivacyConsent(): Boolean {
        return !hasPrivacyPolicyAccepted() || !hasVideoProcessingConsent()
    }

    /**
     * Clear all privacy data (for testing or user request)
     */
    fun clearAllPrivacyData() {
        sharedPrefs.edit().clear().apply()
        updateConsentStatus()
    }

    /**
     * Export privacy data for user (GDPR compliance)
     */
    fun exportPrivacyData(): PrivacyDataExport {
        return PrivacyDataExport(
            videoProcessingConsent = hasVideoProcessingConsent(),
            privacyPolicyAccepted = hasPrivacyPolicyAccepted(),
            consentTimestamp = getConsentTimestamp(),
            privacyPolicyVersion = sharedPrefs.getInt(KEY_PRIVACY_POLICY_VERSION, 0)
        )
    }

    private fun updateConsentStatus() {
        val status = when {
            !hasPrivacyPolicyAccepted() -> PrivacyConsentStatus.PRIVACY_POLICY_REQUIRED
            !hasVideoProcessingConsent() -> PrivacyConsentStatus.VIDEO_CONSENT_REQUIRED
            else -> PrivacyConsentStatus.ALL_GRANTED
        }
        _privacyConsentStatus.value = status
    }

    enum class PrivacyConsentStatus {
        PRIVACY_POLICY_REQUIRED,
        VIDEO_CONSENT_REQUIRED,
        ALL_GRANTED
    }

    data class PrivacyDataExport(
        val videoProcessingConsent: Boolean,
        val privacyPolicyAccepted: Boolean,
        val consentTimestamp: Long,
        val privacyPolicyVersion: Int
    )
}