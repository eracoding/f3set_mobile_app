package com.example.aivideoanalyzer.presentation.ui.upload

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.example.aivideoanalyzer.R
import com.example.aivideoanalyzer.databinding.ItemRecentVideoBinding
import com.example.aivideoanalyzer.domain.model.Video
import com.example.aivideoanalyzer.domain.model.VideoStatus
import java.text.SimpleDateFormat
import java.util.*

class RecentVideosAdapter(
    private val onVideoClick: (Video) -> Unit
) : ListAdapter<Video, RecentVideosAdapter.VideoViewHolder>(VideoDiffCallback()) {

    private val dateFormat = SimpleDateFormat("MMM dd, HH:mm", Locale.getDefault())

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VideoViewHolder {
        val binding = ItemRecentVideoBinding.inflate(
            LayoutInflater.from(parent.context),
            parent,
            false
        )
        return VideoViewHolder(binding)
    }

    override fun onBindViewHolder(holder: VideoViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    inner class VideoViewHolder(
        private val binding: ItemRecentVideoBinding
    ) : RecyclerView.ViewHolder(binding.root) {

        init {
            binding.root.setOnClickListener {
                val position = adapterPosition
                if (position != RecyclerView.NO_POSITION) {
                    onVideoClick(getItem(position))
                }
            }
        }

        fun bind(video: Video) {
            binding.videoName.text = video.name
            binding.uploadDate.text = "Uploaded ${dateFormat.format(video.uploadDate)}"
            binding.videoSize.text = formatFileSize(video.size)

            // Show status with appropriate styling
            binding.statusText.text = getStatusText(video.status)
            binding.statusText.setTextColor(getStatusColor(video.status))

            // Set status icon
            binding.statusIcon.setImageResource(getStatusIcon(video.status))
            binding.statusIcon.setColorFilter(getStatusColor(video.status))
        }

        private fun formatFileSize(size: Long): String {
            val mb = size / (1024.0 * 1024.0)
            return String.format("%.1f MB", mb)
        }

        private fun getStatusText(status: VideoStatus): String {
            return when (status) {
                VideoStatus.UPLOADED -> "Ready for Analysis"
                VideoStatus.PREPROCESSING -> "Preparing..."
                VideoStatus.PROCESSING -> "Analyzing..."
                VideoStatus.COMPLETED -> "Analysis Complete"
                VideoStatus.ERROR -> "Analysis Failed"
            }
        }

        private fun getStatusColor(status: VideoStatus): Int {
            val context = binding.root.context
            return when (status) {
                VideoStatus.UPLOADED -> context.getColor(R.color.md_theme_onSurfaceVariant)
                VideoStatus.PREPROCESSING, VideoStatus.PROCESSING -> context.getColor(R.color.video_processing)
                VideoStatus.COMPLETED -> context.getColor(R.color.video_completed)
                VideoStatus.ERROR -> context.getColor(R.color.video_error)
            }
        }

        private fun getStatusIcon(status: VideoStatus): Int {
            return when (status) {
                VideoStatus.UPLOADED -> R.drawable.ic_upload
                VideoStatus.PREPROCESSING, VideoStatus.PROCESSING -> R.drawable.ic_processing
                VideoStatus.COMPLETED -> R.drawable.ic_check
                VideoStatus.ERROR -> R.drawable.ic_refresh
            }
        }
    }

    class VideoDiffCallback : DiffUtil.ItemCallback<Video>() {
        override fun areItemsTheSame(oldItem: Video, newItem: Video): Boolean {
            return oldItem.id == newItem.id
        }

        override fun areContentsTheSame(oldItem: Video, newItem: Video): Boolean {
            return oldItem == newItem
        }
    }
}