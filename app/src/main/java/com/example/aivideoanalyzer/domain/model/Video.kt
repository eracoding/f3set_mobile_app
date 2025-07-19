package com.example.aivideoanalyzer.domain.model

import android.net.Uri
import java.util.Date

data class Video(
    val id: String,
    val uri: Uri,
    val name: String,
    val size: Long,
    val duration: Long,
    val uploadDate: Date,
    val status: VideoStatus
)

enum class VideoStatus {
    UPLOADED,
    PREPROCESSING,
    PROCESSING,
    COMPLETED,
    ERROR
}