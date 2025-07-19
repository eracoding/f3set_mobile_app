package com.example.aivideoanalyzer.domain.usecase

import android.net.Uri
import com.example.aivideoanalyzer.domain.model.Video
import com.example.aivideoanalyzer.domain.repository.VideoRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

class UploadVideoUseCase(
    private val repository: VideoRepository
) {
    operator fun invoke(uri: Uri): Flow<UploadState> = flow {
        emit(UploadState.Uploading(0))

        try {
            // Validate video
            if (!isValidVideo(uri)) {
                emit(UploadState.Error("Invalid video format"))
                return@flow
            }

            // Upload video
            val result = repository.uploadVideo(uri)

            result.fold(
                onSuccess = { video ->
                    emit(UploadState.Success(video))
                },
                onFailure = { error ->
                    emit(UploadState.Error(error.message ?: "Upload failed"))
                }
            )
        } catch (e: Exception) {
            emit(UploadState.Error(e.message ?: "Unknown error"))
        }
    }

    private fun isValidVideo(uri: Uri): Boolean {
        // TODO: Implement video validation
        return true
    }

    sealed class UploadState {
        data class Uploading(val progress: Int) : UploadState()
        data class Success(val video: Video) : UploadState()
        data class Error(val message: String) : UploadState()
    }
}