package com.example.aivideoanalyzer

import android.app.Application
import com.example.aivideoanalyzer.data.local.FileManager
import com.example.aivideoanalyzer.data.repository.VideoRepositoryImpl
import com.example.aivideoanalyzer.domain.repository.VideoRepository
import com.example.aivideoanalyzer.domain.usecase.*
import com.example.aivideoanalyzer.ml.F3SetInferenceManager
import com.example.aivideoanalyzer.ml.F3SetVideoProcessor
import com.example.aivideoanalyzer.presentation.privacy.PrivacyManager
import com.google.android.material.color.DynamicColors

class AIVideoAnalyzerApplication : Application() {

    // Singletons
    lateinit var fileManager: FileManager
    lateinit var videoRepository: VideoRepository
    lateinit var f3setInferenceManager: F3SetInferenceManager
    lateinit var f3setVideoProcessor: F3SetVideoProcessor
    lateinit var privacyManager: PrivacyManager

    // Use cases
    lateinit var uploadVideoUseCase: UploadVideoUseCase
    lateinit var processVideoUseCase: ProcessVideoUseCase
    lateinit var generateReportUseCase: GenerateReportUseCase

    companion object {
        @Volatile
        private var INSTANCE: AIVideoAnalyzerApplication? = null
    }

    override fun onCreate() {
        super.onCreate()
        INSTANCE = this

        // Apply dynamic color theming
        DynamicColors.applyToActivitiesIfAvailable(this)

        // Initialize dependencies
        initializeDependencies()
    }

    private fun initializeDependencies() {
        // Core components
        fileManager = FileManager(this)
        privacyManager = PrivacyManager(this)
        videoRepository = VideoRepositoryImpl(this, fileManager)

        // F3Set components
        f3setInferenceManager = F3SetInferenceManager(this)
        f3setVideoProcessor = F3SetVideoProcessor(this, f3setInferenceManager)

        // Use cases
        uploadVideoUseCase = UploadVideoUseCase(videoRepository)
        processVideoUseCase = ProcessVideoUseCase(
            videoRepository,
            f3setInferenceManager,
            f3setVideoProcessor
        )
        generateReportUseCase = GenerateReportUseCase()

    }
}