package com.example.aivideoanalyzer.presentation.privacy

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.example.aivideoanalyzer.AIVideoAnalyzerApplication
import kotlinx.coroutines.launch

class PrivacyViewModel(application: Application) : AndroidViewModel(application) {

    private val app = application as AIVideoAnalyzerApplication
    private val privacyManager = app.privacyManager

    private val _consentResult = MutableLiveData<Boolean>()
    val consentResult: LiveData<Boolean> = _consentResult

    private val _privacySettings = MutableLiveData<PrivacySettings>()
    val privacySettings: LiveData<PrivacySettings> = _privacySettings

    val privacyConsentStatus = privacyManager.privacyConsentStatus

    init {
        loadPrivacySettings()
    }

    /**
     * Grant all necessary consents
     */
    fun grantConsent() {
        viewModelScope.launch {
            try {
                privacyManager.acceptPrivacyPolicy()
                privacyManager.grantVideoProcessingConsent()
                privacyManager.markFirstLaunchCompleted()

                _consentResult.value = true
                loadPrivacySettings()
            } catch (e: Exception) {
                _consentResult.value = false
            }
        }
    }

    /**
     * Check if privacy consent is needed
     */
    fun needsPrivacyConsent(): Boolean {
        return privacyManager.needsPrivacyConsent()
    }

    /**
     * Revoke video processing consent
     */
    fun revokeVideoProcessingConsent() {
        viewModelScope.launch {
            privacyManager.revokeVideoProcessingConsent()
            loadPrivacySettings()
        }
    }

    /**
     * Load current privacy settings
     */
    private fun loadPrivacySettings() {
        val settings = PrivacySettings(
            hasVideoProcessingConsent = privacyManager.hasVideoProcessingConsent(),
            hasPrivacyPolicyAccepted = privacyManager.hasPrivacyPolicyAccepted(),
            consentTimestamp = privacyManager.getConsentTimestamp(),
            isFirstLaunch = privacyManager.isFirstLaunch()
        )
        _privacySettings.value = settings
    }

    /**
     * Export privacy data for user
     */
    fun exportPrivacyData(): PrivacyManager.PrivacyDataExport {
        return privacyManager.exportPrivacyData()
    }

    /**
     * Clear all privacy data (user request)
     */
    fun clearAllPrivacyData() {
        viewModelScope.launch {
            privacyManager.clearAllPrivacyData()
            loadPrivacySettings()
        }
    }

    /**
     * Get data usage summary for user
     */
    fun getDataUsageSummary(): DataUsageSummary {
        return DataUsageSummary(
            videosProcessedLocally = true,
            dataSharedWithThirdParties = false,
            dataStoredInCloud = false,
            dataUsedForTraining = false,
            temporaryFilesAutoDeleted = true,
            networkUsageForAnalysis = false
        )
    }

    data class PrivacySettings(
        val hasVideoProcessingConsent: Boolean,
        val hasPrivacyPolicyAccepted: Boolean,
        val consentTimestamp: Long,
        val isFirstLaunch: Boolean
    )

    data class DataUsageSummary(
        val videosProcessedLocally: Boolean,
        val dataSharedWithThirdParties: Boolean,
        val dataStoredInCloud: Boolean,
        val dataUsedForTraining: Boolean,
        val temporaryFilesAutoDeleted: Boolean,
        val networkUsageForAnalysis: Boolean
    )
}