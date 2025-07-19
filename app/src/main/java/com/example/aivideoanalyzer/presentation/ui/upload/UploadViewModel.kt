package com.example.aivideoanalyzer.presentation.ui.upload

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.example.aivideoanalyzer.AIVideoAnalyzerApplication
import com.example.aivideoanalyzer.domain.model.Video
import com.example.aivideoanalyzer.domain.usecase.UploadVideoUseCase
import kotlinx.coroutines.launch

class UploadViewModel(application: Application) : AndroidViewModel(application) {

    sealed class UploadState {
        object Idle : UploadState()
        data class Uploading(val progress: Int) : UploadState()
        data class Success(val video: Video) : UploadState()
        data class Error(val message: String) : UploadState()
    }

    private val _uploadState = MutableLiveData<UploadState>(UploadState.Idle)
    val uploadState: LiveData<UploadState> = _uploadState

    private val _selectedVideo = MutableLiveData<Video?>()
    val selectedVideo: LiveData<Video?> = _selectedVideo

    private val _uploadedVideos = MutableLiveData<List<Video>>(emptyList())
    val uploadedVideos: LiveData<List<Video>> = _uploadedVideos

    private val _showSuccessMessage = MutableLiveData<String?>()
    val showSuccessMessage: LiveData<String?> = _showSuccessMessage

    private val app = application as AIVideoAnalyzerApplication
    private val uploadUseCase = app.uploadVideoUseCase

    init {
        // Load existing videos when the fragment is opened
        loadExistingVideos()
    }

    fun onVideoSelected(uri: Uri) {
        viewModelScope.launch {
            _uploadState.value = UploadState.Uploading(0)

            try {
                uploadUseCase(uri).collect { state ->
                    when (state) {
                        is UploadVideoUseCase.UploadState.Uploading -> {
                            _uploadState.value = UploadState.Uploading(state.progress)
                        }
                        is UploadVideoUseCase.UploadState.Success -> {
                            _uploadState.value = UploadState.Success(state.video)
                            _selectedVideo.value = state.video

                            // Update uploaded videos list
                            val currentList = _uploadedVideos.value ?: emptyList()
                            _uploadedVideos.value = currentList + state.video

                            // Show success message instead of auto-navigating
                            _showSuccessMessage.value = "Tennis video uploaded successfully! Go to Processing tab to start F3Set analysis."

                            // Reset to idle state after showing success
                            kotlinx.coroutines.delay(3000)
                            _uploadState.value = UploadState.Idle
                            _selectedVideo.value = null
                        }
                        is UploadVideoUseCase.UploadState.Error -> {
                            _uploadState.value = UploadState.Error(state.message)
                        }
                    }
                }
            } catch (e: Exception) {
                _uploadState.value = UploadState.Error(e.message ?: "Upload failed")
            }
        }
    }

    fun retryUpload() {
        _selectedVideo.value?.uri?.let { uri ->
            onVideoSelected(uri)
        }
    }

    fun clearSelection() {
        _selectedVideo.value = null
        _uploadState.value = UploadState.Idle
    }

    fun clearSuccessMessage() {
        _showSuccessMessage.value = null
    }

    private fun loadExistingVideos() {
        viewModelScope.launch {
            app.videoRepository.getAllVideos().fold(
                onSuccess = { videos ->
                    _uploadedVideos.value = videos
                },
                onFailure = {
                    // Handle error silently for now
                }
            )
        }
    }

    fun refreshVideos() {
        loadExistingVideos()
    }
}