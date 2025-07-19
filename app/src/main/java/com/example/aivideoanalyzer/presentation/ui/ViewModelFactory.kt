package com.example.aivideoanalyzer.presentation.ui

import android.app.Application
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.aivideoanalyzer.presentation.privacy.PrivacyViewModel
import com.example.aivideoanalyzer.presentation.ui.main.MainViewModel
import com.example.aivideoanalyzer.presentation.ui.processing.ProcessingViewModel
import com.example.aivideoanalyzer.presentation.ui.results.ResultsViewModel
import com.example.aivideoanalyzer.presentation.ui.upload.UploadViewModel

class ViewModelFactory(
    private val application: Application
) : ViewModelProvider.Factory {

    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        return when {
            modelClass.isAssignableFrom(MainViewModel::class.java) -> {
                MainViewModel(application) as T
            }
            modelClass.isAssignableFrom(UploadViewModel::class.java) -> {
                UploadViewModel(application) as T
            }
            modelClass.isAssignableFrom(ProcessingViewModel::class.java) -> {
                ProcessingViewModel(application) as T
            }
            modelClass.isAssignableFrom(ResultsViewModel::class.java) -> {
                ResultsViewModel(application) as T
            }
            modelClass.isAssignableFrom(PrivacyViewModel::class.java) -> {
                PrivacyViewModel(application) as T
            }
            else -> throw IllegalArgumentException("Unknown ViewModel class: ${modelClass.name}")
        }
    }
}