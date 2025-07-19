package com.example.aivideoanalyzer.data.local

import android.content.Context
import android.net.Uri
import android.os.Environment
import java.io.File
import java.io.FileOutputStream

class FileManager(private val context: Context) {

    private val videosDir: File by lazy {
        File(context.filesDir, "videos").apply {
            if (!exists()) mkdirs()
        }
    }

    private val exportsDir: File by lazy {
        // Use public Documents directory for easier access
        val documentsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
        File(documentsDir, "AIVideoAnalyzer").apply {
            if (!exists()) mkdirs()
        }
    }

    fun saveVideoFile(uri: Uri, videoId: String): File? {
        return try {
            val inputStream = context.contentResolver.openInputStream(uri) ?: return null
            val videoFile = File(videosDir, "$videoId.mp4")

            FileOutputStream(videoFile).use { output ->
                inputStream.copyTo(output)
            }

            inputStream.close()
            videoFile
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    fun deleteVideoFile(videoId: String): Boolean {
        val videoFile = File(videosDir, "$videoId.mp4")
        return if (videoFile.exists()) {
            videoFile.delete()
        } else {
            true
        }
    }
}