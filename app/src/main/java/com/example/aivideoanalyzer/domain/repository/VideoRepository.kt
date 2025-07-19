package com.example.aivideoanalyzer.domain.repository

import android.net.Uri
import com.example.aivideoanalyzer.domain.model.AnalysisResult
import com.example.aivideoanalyzer.domain.model.Video
import com.example.aivideoanalyzer.domain.model.VideoStatus
import kotlinx.coroutines.flow.Flow

interface VideoRepository {

    // Video management
    suspend fun uploadVideo(uri: Uri): Result<Video>
    suspend fun getVideo(videoId: String): Result<Video>
    suspend fun getAllVideos(): Result<List<Video>>
    suspend fun deleteVideo(videoId: String): Result<Unit>
    suspend fun updateVideoStatus(videoId: String, status: VideoStatus): Result<Unit>

    // Video processing
    suspend fun processVideo(videoId: String): Flow<ProcessingProgress>
    suspend fun cancelProcessing(videoId: String): Result<Unit>
    suspend fun retryProcessing(videoId: String): Result<Unit>

    // Analysis results
    suspend fun saveAnalysisResult(result: AnalysisResult): Result<Unit>
    suspend fun getAnalysisResult(videoId: String): Result<AnalysisResult>
    suspend fun getAllAnalysisResults(): Result<List<AnalysisResult>>
    suspend fun deleteAnalysisResult(videoId: String): Result<Unit>

    // Export functionality
    suspend fun exportToExcel(result: AnalysisResult, filePath: String): Result<String>
    suspend fun exportToCsv(result: AnalysisResult, filePath: String): Result<String>
    suspend fun exportToPdf(result: AnalysisResult, filePath: String): Result<String>

    // Observables
    fun observeVideos(): Flow<List<Video>>
    fun observeAnalysisResults(): Flow<List<AnalysisResult>>
}

data class ProcessingProgress(
    val stage: ProcessingStage,
    val progress: Int,
    val message: String
)

enum class ProcessingStage {
    UPLOADING,
    PREPROCESSING,
    INFERENCING,
    POST_PROCESSING,
    GENERATING_REPORT,
    COMPLETED
}