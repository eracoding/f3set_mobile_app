package com.example.aivideoanalyzer.domain.model

import java.util.Date

data class AnalysisResult(
    val videoId: String,
    val timestamp: Date,
    val frames: List<FrameAnalysis>,
    val summary: String,
    val confidence: Float
)

data class FrameAnalysis(
    val frameNumber: Int,
    val timestamp: Long,
    val detections: List<Detection>,
    val confidence: Float
)

data class Detection(
    val label: String,
    val confidence: Float,
    val boundingBox: BoundingBox?
)

data class BoundingBox(
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float
)