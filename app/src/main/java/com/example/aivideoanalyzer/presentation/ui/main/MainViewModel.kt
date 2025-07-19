package com.example.aivideoanalyzer.presentation.ui.main

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.example.aivideoanalyzer.AIVideoAnalyzerApplication
import kotlinx.coroutines.launch

class MainViewModel(application: Application) : AndroidViewModel(application) {

    sealed class AppState {
        object Loading : AppState()
        object Ready : AppState()
        data class Error(val message: String) : AppState()
    }

    private val _appState = MutableLiveData<AppState>()
    val appState: LiveData<AppState> = _appState

    private val _currentVideoCount = MutableLiveData(0)

    private val app = application as AIVideoAnalyzerApplication

    init {
        initializeApp()
    }

    private fun initializeApp() {
        viewModelScope.launch {
            _appState.value = AppState.Loading

            try {
                // Load saved data
                val videos = app.videoRepository.getAllVideos().getOrNull()
                _currentVideoCount.value = videos?.size ?: 0

                _appState.value = AppState.Ready
            } catch (e: Exception) {
                _appState.value = AppState.Error(e.message ?: "Unknown error occurred")
            }
        }
    }
}
