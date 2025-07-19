# F3Set Tennis AI Video Analyzer

<div align="center">
  <!-- <img src="https://via.placeholder.com/128x128/006590/FFFFFF?text=🎾" alt="F3Set Logo" width="128"/> -->
  
  [![Android](https://img.shields.io/badge/Platform-Android-green.svg)](https://android.com)
  [![Kotlin](https://img.shields.io/badge/Language-Kotlin-blue.svg)](https://kotlinlang.org)
  [![PyTorch Mobile](https://img.shields.io/badge/AI-PyTorch%20Mobile-orange.svg)](https://pytorch.org/mobile)
  [![API](https://img.shields.io/badge/API-26%2B-brightgreen.svg)](https://android-arsenal.com/api?level=26)
  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
</div>

## Overview

F3Set Tennis AI Video Analyzer is a privacy-first Android application that performs **real-time tennis action recognition** using on-device machine learning. Built with the F3Set (Fine-grained Tennis Shot Recognition) model, the app analyzes tennis videos locally without any data leaving your device.

### Key Features

- **Advanced Tennis Action Recognition**: Detects and classifies 29 different tennis actions including serves, forehands, backhands, volleys, and more
- **Complete Privacy**: All video processing happens locally on your device - no cloud uploads
- **Mobile-Optimized**: Custom PyTorch Mobile implementation optimized for Android devices
- **Detailed Analytics**: Frame-by-frame analysis with confidence scores and bounding boxes
- **Export Capabilities**: Export results in HTML, CSV, and JSON formats
- **Modern UI**: Material Design 3 interface with comprehensive privacy controls

## What Makes F3Set Special

### Advanced Tennis Action Classification
The F3Set model recognizes 29 distinct tennis actions across multiple categories:

- **Court Position**: near, far, deuce, middle, ad
- **Shot Types**: serve, return, stroke, forehand, backhand, groundstroke, slice, volley, smash, drop, lob
- **Ball Direction**: T, B, W, CC, DL, DM, II, IO (trajectory patterns)
- **Game Context**: approach, in, winner, forced-error, unforced-error

### Privacy-by-Design Architecture
- **Local Processing**: PyTorch Mobile models run entirely on-device
- **Zero Cloud Dependencies**: No internet required for video analysis
- **Automatic Cleanup**: Temporary files are automatically deleted
- **User Control**: Complete data ownership and deletion capabilities

## Getting Started

### Prerequisites

- **Android 8.0 (API level 26) or higher**
- **4GB+ RAM** recommended for optimal performance
- **ARM64 processor** (most modern Android devices)

### Installation

1. **Download the APK**
   Option 1: Download from google drive
   [Visit](https://drive.google.com/drive/folders/1gOXjCR5dbeLFJ7WWPnJ5hoIL0nYToXlw?usp=sharing)

2. **Install on Device**
   ```bash
   adb install apk-release.apk
   ```

3. **Grant Permissions**
   - Storage access for video files
   - Optional: Camera access for recording

### Quick Start Guide

1. **Upload Video**: Select a tennis video (MP4, MOV, AVI up to 500MB)
2. **Grant Consent**: Accept privacy policy and local processing consent
3. **Start Analysis**: The F3Set model will analyze your video frame-by-frame
4. **View Results**: Explore detected tennis actions with timestamps and confidence scores
5. **Export**: Save results as HTML reports, CSV data, or JSON for further analysis

## 🏗️ Architecture

### Core Components

```
F3Set Tennis Analyzer
├── ML Engine (PyTorch Mobile)
│   ├── F3SetInferenceManager - Model loading and inference
│   ├── F3SetVideoProcessor - Video frame processing
│   └── TensorFlow Operations - Optimized mobile operations
├── Video Pipeline
│   ├── Frame Extraction - Precise timing and quality
│   ├── Preprocessing - Normalization and resizing
│   └── Batch Processing - Efficient memory management
├── Analysis Engine
│   ├── Detection Aggregation - Multi-frame voting
│   ├── Tennis Rules Engine - Sport-specific logic
│   └── Confidence Scoring - Reliability metrics
├── Privacy Layer
│   ├── Local Storage - Encrypted on-device data
│   ├── Consent Management - GDPR-compliant controls
│   └── Automatic Cleanup - Secure file deletion
└── User Interface
    ├── Material Design 3 - Modern, accessible UI
    ├── Real-time Progress - Live processing updates
    └── Export Tools - Multiple output formats
```

### Technical Stack

- **Frontend**: Kotlin, Android Jetpack, Material Design 3
- **ML Framework**: PyTorch Mobile, custom C++ optimizations
- **Architecture**: MVVM with Repository pattern
- **Dependencies**: 
  - AndroidX libraries
  - Navigation Components
  - Coroutines for async processing
  - OpenCSV for data export

## How It Works

### 1. Video Input Processing
```kotlin
// Video frames are extracted at 30 FPS with precise timing
val frames = videoProcessor.extractFrames(videoUri, targetFps = 30)
```

### 2. F3Set Model Inference
```kotlin
// Sliding window approach with 96-frame clips
val clips = frames.chunked(96, stride = 48)
clips.forEach { clip ->
    val result = f3setModel.predict(clip, handInfo)
    detections.addAll(result.shots)
}
```

### 3. Tennis-Specific Post-Processing
```kotlin
// Apply tennis rules and aggregate detections
val finalDetections = applyTennisRules(rawDetections)
val shots = aggregateWithVoting(finalDetections)
```

### 4. Result Generation
- Frame-by-frame action classification
- Temporal shot segmentation
- Confidence scoring and validation
- Export-ready data formatting

## Performance Metrics

### Device Compatibility
| Device Category | Performance | Status |
|----------------|-------------|--------|
| Flagship (2022+) | Excellent | Fully Supported |
| Mid-range (2020+) | Good | Supported |
| Budget (2019+) | Limited | Basic Support |
| Older devices | Minimal | Not Recommended |

## Privacy & Security

### Data Handling Principles

1. **Local-First Architecture**
   - All processing happens on your device
   - No video data ever transmitted
   - No cloud storage or external servers

2. **Transparent Consent**
   - Clear privacy policy with technical details
   - Granular consent controls
   - Easy consent revocation

3. **Secure Storage**
   - Encrypted local file storage
   - Automatic temporary file cleanup
   - User-controlled data deletion

4. **GDPR Compliance**
   - Data portability (export your data)
   - Right to deletion
   - Privacy by design principles

### What Data We DON'T Collect
- ❌ Video files (deleted after processing)
- ❌ Personal information
- ❌ Usage analytics
- ❌ Device identifiers
- ❌ Location data

### What Data We DO Store (Locally Only)
- ✅ Analysis results (user controlled)
- ✅ App preferences
- ✅ Consent records
- ✅ Exported files (in user Documents)

## Export Formats

### HTML Report
Rich, interactive reports with:
- Executive summary with key metrics
- Shot timeline with visual indicators
- Frame-by-frame breakdowns
- Confidence scoring charts

### CSV Data
Machine-readable format with:
- Frame numbers and timestamps
- Action classifications
- Confidence scores
- Bounding box coordinates

### JSON Format
Developer-friendly structure:
```json
{
  "videoId": "tennis_match_2024.mp4",
  "analysisType": "F3Set Tennis Action Recognition",
  "shots": [
    {
      "frameNumber": 150,
      "timestamp": 5000,
      "detections": [
        {
          "action": "serve",
          "confidence": 0.94,
        }
      ]
    }
  ]
}
```

## Development

### Building from Source

1. **Clone Repository**
   ```bash
   git clone https://github.com/eracoding/f3set_mobile_app.git
   cd f3set_mobile_app
   ```

2. **Setup Android Studio**
   - Install Android Studio Arctic Fox or later
   - SDK API 34 (Android 14)
   - NDK for PyTorch Mobile

3. **Configure Model Files**
   ```bash
   # Place F3Set model in assets
   mkdir -p app/src/main/assets/pad/
   cp model_scripted.pt app/src/main/assets/pad/
   ```

4. **Build APK**
   ```bash
   ./gradlew assembleDebug
   # Or for release
   ./gradlew assembleRelease
   ```

### Project Structure
```
app/
├── src/main/
│   ├── java/com/example/aivideoanalyzer/
│   │   ├── ml/                     # F3Set ML components
│   │   ├── presentation/           # UI layers
│   │   ├── domain/                 # Business logic
│   │   └── data/                   # Data layer
│   ├── assets/pad/                 # F3Set model files
│   └── res/                        # Resources
├── build.gradle                    # Dependencies
└── proguard-rules.pro             # Code obfuscation
```

### Key Dependencies
```gradle
// Core Android
implementation 'androidx.core:core-ktx:1.12.0'
implementation 'androidx.lifecycle:lifecycle-viewmodel-ktx:2.7.0'

// UI
implementation 'com.google.android.material:material:1.11.0'
implementation 'androidx.navigation:navigation-fragment-ktx:2.7.6'

// ML
implementation 'org.pytorch:pytorch_android:1.12.2'
implementation 'org.pytorch:pytorch_android_torchvision:1.12.2'

// Data
implementation 'com.opencsv:opencsv:5.8'
```

## Testing

### Running Tests
```bash
# Unit tests
./gradlew test

# Instrumented tests
./gradlew connectedAndroidTest

# F3Set model tests
./gradlew testDebugUnitTest --tests "*F3SetTest*"
```

### Model Validation
The app includes built-in model validation:
- Expected test frames: [73, 103, 144, 189, 223, 273, 304]
- Confidence thresholds and detection criteria
- Memory usage monitoring

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Areas for Contribution
- **Sport-Specific Features**: Additional tennis metrics and analytics
- **Performance**: Model optimization and inference speed improvements  
- **UI/UX**: Enhanced visualizations and user experience
- **Privacy**: Additional privacy features and security enhancements
- **Export**: New export formats and integrations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **F3Set Research**: Based on the F3Set tennis action recognition dataset and methodology
- **PyTorch Mobile**: For providing excellent on-device ML capabilities
- **Material Design**: For comprehensive design system and components
- **Tennis Community**: For feedback and testing during development

## Support

### Getting Help
- **Documentation**: Check our [Technical Documentation](TECHNICAL.md)
- **Bug Reports**: [GitHub Issues](https://github.com/eracoding/f3set_mobile_app/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/eracoding/f3set_mobile_app/discussions)

### FAQ

**Q: Why does processing take so long?**
A: F3Set analysis is computationally intensive. Processing time depends on video length and device performance. A 1-minute video typically takes 30-60 seconds on modern devices.

**Q: Can I use this for other sports?**
A: Currently optimized for tennis. The F3Set model is specifically trained on tennis actions, but the architecture could be adapted for other sports.

**Q: Does this work offline?**
A: Yes! After initial installation, the app works completely offline. No internet required for video analysis.

**Q: What video formats are supported?**
A: MP4, MOV, AVI, and MKV formats up to 500MB. 1080p resolution recommended for best results.

---

<div align="center">
  <p><strong>🎾 Serve up better tennis analysis with F3Set! 🎾</strong></p>
  <p>Made with ❤️ for the tennis community</p>
</div>
