package com.example.aivideoanalyzer.presentation.ui.upload

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.aivideoanalyzer.databinding.FragmentUploadBinding
import com.example.aivideoanalyzer.presentation.ui.ViewModelFactory
import com.google.android.material.snackbar.Snackbar

class UploadFragment : Fragment() {

    private var _binding: FragmentUploadBinding? = null
    private val binding get() = _binding!!

    private val viewModel: UploadViewModel by viewModels {
        ViewModelFactory(requireActivity().application)
    }

    private lateinit var recentVideosAdapter: RecentVideosAdapter

    private val videoPickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            viewModel.onVideoSelected(it)
        }
    }

    private val galleryLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            result.data?.data?.let { uri ->
                viewModel.onVideoSelected(uri)
            }
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentUploadBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        setupUI()
        setupRecentVideos()
        observeViewModel()
    }

    override fun onResume() {
        super.onResume()
        // Refresh videos when returning to this fragment
        viewModel.refreshVideos()
    }

    private fun setupUI() {
        binding.uploadButton.setOnClickListener {
            showVideoSourceDialog()
        }

        binding.dragDropArea.setOnClickListener {
            showVideoSourceDialog()
        }

        binding.swipeRefreshLayout.setOnRefreshListener {
            viewModel.refreshVideos()
        }
    }

    private fun setupRecentVideos() {
        recentVideosAdapter =
            RecentVideosAdapter { video ->
                showMessage("Video already uploaded. Check Processing tab to analyze it.")
            }

        binding.recentVideosRecyclerView.apply {
            layoutManager = LinearLayoutManager(context)
            adapter = recentVideosAdapter
        }
    }

    private fun showVideoSourceDialog() {
        val options = arrayOf("From Files", "From Gallery", "Record Video")

        AlertDialog.Builder(requireContext())
            .setTitle("Select Tennis Video Source")
            .setItems(options) { _, which ->
                when (which) {
                    0 -> videoPickerLauncher.launch("video/*")
                    1 -> openGallery()
                    2 -> showMessage("Video recording feature coming soon!")
                }
            }
            .show()
    }

    private fun openGallery() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
        galleryLauncher.launch(intent)
    }

    private fun observeViewModel() {
        viewModel.uploadState.observe(viewLifecycleOwner) { state ->
            when (state) {
                is UploadViewModel.UploadState.Idle -> {
                    binding.progressIndicator.visibility = View.GONE
                    binding.uploadButton.isEnabled = true
                    binding.progressText.visibility = View.GONE
                }
                is UploadViewModel.UploadState.Uploading -> {
                    binding.progressIndicator.visibility = View.VISIBLE
                    binding.uploadButton.isEnabled = false
                    binding.progressText.visibility = View.VISIBLE
                    binding.progressText.text = "Uploading tennis video... ${state.progress}%"
                    binding.progressIndicator.progress = state.progress
                }
                is UploadViewModel.UploadState.Success -> {
                    binding.progressIndicator.visibility = View.GONE
                    binding.uploadButton.isEnabled = true
                    binding.progressText.visibility = View.GONE
                    // Success message is handled by showSuccessMessage observer
                }
                is UploadViewModel.UploadState.Error -> {
                    binding.progressIndicator.visibility = View.GONE
                    binding.uploadButton.isEnabled = true
                    binding.progressText.visibility = View.GONE
                    showError(state.message)
                }
            }
        }

        viewModel.selectedVideo.observe(viewLifecycleOwner) { video ->
            video?.let {
                binding.selectedVideoName.text = video.name
                binding.selectedVideoSize.text = formatFileSize(video.size)
                binding.selectedVideoCard.visibility = View.VISIBLE
            } ?: run {
                binding.selectedVideoCard.visibility = View.GONE
            }
        }

        viewModel.uploadedVideos.observe(viewLifecycleOwner) { videos ->
            binding.swipeRefreshLayout.isRefreshing = false
            if (videos.isNotEmpty()) {
                binding.recentVideosSection.visibility = View.VISIBLE
                binding.recentVideosCount.text = "${videos.size} video(s) uploaded"
                recentVideosAdapter.submitList(videos.takeLast(5)) // Show last 5 uploads
            } else {
                binding.recentVideosSection.visibility = View.GONE
            }
        }

        viewModel.showSuccessMessage.observe(viewLifecycleOwner) { message ->
            message?.let {
                showSuccessWithAction(it)
                viewModel.clearSuccessMessage()
            }
        }
    }

    private fun showSuccessWithAction(message: String) {
        Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG)
            .setAction("Go to Processing") {
                // Navigate to processing tab
                requireActivity().findViewById<com.google.android.material.bottomnavigation.BottomNavigationView>(
                    com.example.aivideoanalyzer.R.id.bottom_navigation
                )?.selectedItemId = com.example.aivideoanalyzer.R.id.processingFragment
            }
            .show()
    }

    private fun showError(message: String) {
        Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG)
            .setAction("Retry") {
                viewModel.retryUpload()
            }
            .show()
    }

    private fun showMessage(message: String) {
        Snackbar.make(binding.root, message, Snackbar.LENGTH_SHORT).show()
    }

    private fun formatFileSize(size: Long): String {
        val kb = size / 1024.0
        val mb = kb / 1024.0
        return when {
            mb >= 1 -> String.format("%.2f MB", mb)
            kb >= 1 -> String.format("%.2f KB", kb)
            else -> "$size bytes"
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}