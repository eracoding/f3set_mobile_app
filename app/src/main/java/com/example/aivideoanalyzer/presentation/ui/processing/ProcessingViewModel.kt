package com.example.aivideoanalyzer.presentation.ui.processing

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.example.aivideoanalyzer.AIVideoAnalyzerApplication
import com.example.aivideoanalyzer.domain.model.Video
import com.example.aivideoanalyzer.domain.model.VideoStatus
import com.example.aivideoanalyzer.domain.repository.ProcessingStage
import com.example.aivideoanalyzer.domain.repository.VideoRepository
import com.example.aivideoanalyzer.domain.usecase.ProcessVideoUseCase
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch

class ProcessingViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "ProcessingViewModel"
    }

    sealed class ProcessingState {
        object Idle : ProcessingState()
        data class Processing(val videoName: String, val progress: Int, val message: String = "") : ProcessingState()
        data class Completed(val videoId: String) : ProcessingState()
        data class Error(val message: String) : ProcessingState()
    }

    data class Statistics(
        val total: Int = 0,
        val processing: Int = 0,
        val completed: Int = 0,
        val error: Int = 0
    )

    private val _processingVideos = MutableLiveData<List<Video>>(emptyList())
    val processingVideos: LiveData<List<Video>> = _processingVideos

    private val _processingState = MutableLiveData<ProcessingState>(ProcessingState.Idle)
    val processingState: LiveData<ProcessingState> = _processingState

    private val _statistics = MutableLiveData<Statistics>()
    val statistics: LiveData<Statistics> = _statistics

    private var currentFilter: String = "All"
    private var allVideos: List<Video> = emptyList()

    private val app = application as AIVideoAnalyzerApplication
    private val processVideoUseCase: ProcessVideoUseCase = app.processVideoUseCase
    private val videoRepository: VideoRepository = app.videoRepository

    init {
        observeVideos()
        loadVideos()
    }

    private fun observeVideos() {
        viewModelScope.launch {
            videoRepository.observeVideos().collect { videos ->
                allVideos = videos
                applyFilter()
                updateStatistics(videos)
            }
        }
    }

    private fun loadVideos() {
        viewModelScope.launch {
            videoRepository.getAllVideos().fold(
                onSuccess = { videos ->
                    allVideos = videos
                    applyFilter()
                    updateStatistics(videos)
                },
                onFailure = { error ->
//                    Log.e(TAG, "Failed to load videos", error)
                }
            )
        }
    }

    fun onVideoSelected(video: Video) {
        // Handle video selection - could navigate to detail view
//        Log.d(TAG, "Video selected: ${video.name}")
    }

    fun toggleProcessing(video: Video) {
        viewModelScope.launch {
            when (video.status) {
                VideoStatus.UPLOADED -> startProcessing(video)
                VideoStatus.ERROR -> retryProcessing(video)
                VideoStatus.PROCESSING, VideoStatus.PREPROCESSING -> pauseProcessing(video)
                else -> {
//                    Log.d(TAG, "No action for video status: ${video.status}")
                }
            }
        }
    }

    private fun startProcessing(video: Video) {
        viewModelScope.launch {
//            Log.d(TAG, "Starting processing for video: ${video.name}")

            try {
                processVideoUseCase(video.id)
                    .catch { error ->
//                        Log.e(TAG, "Processing error for video: ${video.name}", error)
                        _processingState.value = ProcessingState.Error(
                            error.message ?: "Processing failed"
                        )
                        videoRepository.updateVideoStatus(video.id, VideoStatus.ERROR)
                    }
                    .collect { progress ->
//                        Log.d(TAG, "Progress update: ${progress.stage} - ${progress.progress}% - ${progress.message}")

                        when (progress.stage) {
                            ProcessingStage.UPLOADING -> {
                                _processingState.value = ProcessingState.Processing(
                                    videoName = video.name,
                                    progress = progress.progress,
                                    message = "Uploading..."
                                )
                            }
                            ProcessingStage.PREPROCESSING -> {
                                _processingState.value = ProcessingState.Processing(
                                    videoName = video.name,
                                    progress = progress.progress,
                                    message = "Preprocessing: ${progress.message}"
                                )
                            }
                            ProcessingStage.INFERENCING -> {
                                _processingState.value = ProcessingState.Processing(
                                    videoName = video.name,
                                    progress = progress.progress,
                                    message = "AI Analysis: ${progress.message}"
                                )
                            }
                            ProcessingStage.POST_PROCESSING -> {
                                _processingState.value = ProcessingState.Processing(
                                    videoName = video.name,
                                    progress = progress.progress,
                                    message = "Post-processing: ${progress.message}"
                                )
                            }
                            ProcessingStage.GENERATING_REPORT -> {
                                _processingState.value = ProcessingState.Processing(
                                    videoName = video.name,
                                    progress = progress.progress,
                                    message = "Generating Report: ${progress.message}"
                                )
                            }
                            ProcessingStage.COMPLETED -> {
//                                Log.d(TAG, "Processing completed for video: ${video.name}")
                                _processingState.value = ProcessingState.Completed(video.id)

                                // Reset to idle after a delay
                                kotlinx.coroutines.delay(3000)
                                _processingState.value = ProcessingState.Idle
                            }
                        }
                    }

            } catch (e: Exception) {
//                Log.e(TAG, "Processing failed for video: ${video.name}", e)
                _processingState.value = ProcessingState.Error(e.message ?: "Processing failed")
                videoRepository.updateVideoStatus(video.id, VideoStatus.ERROR)
            }
        }
    }

    private fun retryProcessing(video: Video) {
        viewModelScope.launch {
//            Log.d(TAG, "Retrying processing for video: ${video.name}")
            processVideoUseCase.retry(video.id)
            startProcessing(video)
        }
    }

    private fun pauseProcessing(video: Video) {
        viewModelScope.launch {
//            Log.d(TAG, "Pausing processing for video: ${video.name}")
            processVideoUseCase.cancel(video.id)
            _processingState.value = ProcessingState.Idle
        }
    }

    fun filterByStatus(status: String) {
        currentFilter = status
        applyFilter()
    }

    private fun applyFilter() {
        val filteredVideos = when (currentFilter) {
            "All" -> allVideos
            "Processing" -> allVideos.filter {
                it.status == VideoStatus.PREPROCESSING || it.status == VideoStatus.PROCESSING
            }
            "Completed" -> allVideos.filter { it.status == VideoStatus.COMPLETED }
            "Error" -> allVideos.filter { it.status == VideoStatus.ERROR }
            else -> allVideos
        }
        _processingVideos.value = filteredVideos
    }

    private fun updateStatistics(videos: List<Video>) {
        val stats = Statistics(
            total = videos.size,
            processing = videos.count {
                it.status == VideoStatus.PREPROCESSING || it.status == VideoStatus.PROCESSING
            },
            completed = videos.count { it.status == VideoStatus.COMPLETED },
            error = videos.count { it.status == VideoStatus.ERROR }
        )
        _statistics.value = stats
//        Log.d(TAG, "Statistics updated: $stats")
    }
}