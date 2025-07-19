plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.aivideoanalyzer"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.aivideoanalyzer"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Add these for better APK optimization
        vectorDrawables {
            useSupportLibrary = true
        }

        // Optimize for different architectures
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
    }

    // Signing configurations for release builds
    signingConfigs {
        create("release") {
            // Read from environment variables or local.properties
            val keystoreFile = file("../keystore/release.keystore")
            if (keystoreFile.exists()) {
                storeFile = keystoreFile
                storePassword = System.getenv("KEYSTORE_PASSWORD") ?: project.findProperty("KEYSTORE_PASSWORD") as String? ?: "defaultpassword"
                keyAlias = "release"
                keyPassword = System.getenv("KEY_PASSWORD") ?: project.findProperty("KEY_PASSWORD") as String? ?: "defaultpassword"

                // Enable V1 and V2 signature schemes
                enableV1Signing = true
                enableV2Signing = true
                enableV3Signing = true
                enableV4Signing = true
            }
        }
    }

    buildTypes {
        debug {
            applicationIdSuffix = ".debug"
            isMinifyEnabled = false
            versionNameSuffix = "-debug"
        }

        release {
            // TEMPORARILY DISABLE MINIFICATION TO AVOID R8 ISSUES
            isMinifyEnabled = false  // Changed from true to false
            isShrinkResources = false  // Changed from true to false
            isDebuggable = false

            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )

            // Use release signing config if available
            if (signingConfigs.names.contains("release")) {
                signingConfig = signingConfigs.getByName("release")
            }

            // Optimize APK for distribution
            isZipAlignEnabled = true
            isCrunchPngs = true
        }

        // Create a "minified" build type for when you want to test R8
        create("minified") {
            initWith(getByName("release"))
            isMinifyEnabled = true
            isShrinkResources = true
            applicationIdSuffix = ".minified"
            versionNameSuffix = "-minified"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"

        // Add these for better Kotlin optimization
        freeCompilerArgs += listOf(
            "-Xopt-in=kotlin.RequiresOptIn"
        )
    }

    buildFeatures {
        viewBinding = true
        buildConfig = true
    }

    packagingOptions {
        resources {
            excludes += setOf(
                "/META-INF/{AL2.0,LGPL2.1}",
                "/META-INF/DEPENDENCIES",
                "/META-INF/LICENSE",
                "/META-INF/LICENSE.txt",
                "/META-INF/NOTICE",
                "/META-INF/NOTICE.txt",
                "/META-INF/ASL2.0",
                "/META-INF/LGPL2.1",
                "/META-INF/gradle/incremental.annotation.processors"
            )
        }

        // Handle native libraries for PyTorch - IMPORTANT FIX
        jniLibs {
            pickFirsts += setOf(
                "**/libc++_shared.so",
                "**/libjsc.so",
                "**/libfbjni.so",
                "**/libpytorch_jni.so"
            )
        }
    }

    // Configure lint for release builds
    lint {
        checkReleaseBuilds = true
        abortOnError = false
        disable += setOf("MissingTranslation", "ExtraTranslation")
    }

    // Bundle configuration - keep everything in base APK for direct distribution
    bundle {
        language {
            enableSplit = false
        }
        density {
            enableSplit = false
        }
        abi {
            enableSplit = false
        }
    }
}

dependencies {
    // Core Android dependencies
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    implementation("androidx.activity:activity-ktx:1.8.2")
    implementation("androidx.fragment:fragment-ktx:1.6.2")

    // Material Design 3
    implementation("com.google.android.material:material:1.11.0")

    // ConstraintLayout
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // Navigation Component
    implementation("androidx.navigation:navigation-fragment-ktx:2.7.7")
    implementation("androidx.navigation:navigation-ui-ktx:2.7.7")

    // ViewModel and LiveData
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.7.0")
    implementation("androidx.lifecycle:lifecycle-livedata-ktx:2.7.0")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // Splash Screen
    implementation("androidx.core:core-splashscreen:1.0.1")

    // SwipeRefreshLayout
    implementation("androidx.swiperefreshlayout:swiperefreshlayout:1.1.0")

    // PyTorch dependencies for F3Set model
    implementation("org.pytorch:pytorch_android:1.13.1")
    implementation("org.pytorch:pytorch_android_torchvision:1.13.1")

    // CSV export functionality
    implementation("com.opencsv:opencsv:5.7.1")

    // ADD MISSING ANNOTATION DEPENDENCY - FIX FOR R8 ERROR
    implementation("androidx.annotation:annotation:1.7.1")
    compileOnly("javax.annotation:javax.annotation-api:1.3.2")

    // Optional: Add these for better user experience
    implementation("androidx.work:work-runtime-ktx:2.9.0")
    implementation("androidx.preference:preference-ktx:1.2.1")

    // Testing dependencies
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.mockito:mockito-core:5.1.1")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.7.3")

    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
    androidTestImplementation("androidx.test:runner:1.5.2")
    androidTestImplementation("androidx.test:rules:1.5.0")
}

// Custom tasks for deployment
tasks.register("printVersionInfo") {
    doLast {
        println("=================================")
        println("F3Set Tennis Analyzer Build Info")
        println("=================================")
        println("Version Name: ${android.defaultConfig.versionName}")
        println("Version Code: ${android.defaultConfig.versionCode}")
        println("Application ID: ${android.defaultConfig.applicationId}")
        println("Min SDK: ${android.defaultConfig.minSdk}")
        println("Target SDK: ${android.defaultConfig.targetSdk}")
        println("Compile SDK: ${android.compileSdk}")
        println("=================================")
    }
}

tasks.register<Copy>("copyReleaseApk") {
    dependsOn("assembleRelease")

    from("build/outputs/apk/release/")
    into("../distribution/")
    include("*.apk")

    rename { filename ->
        "f3set-tennis-analyzer-v${android.defaultConfig.versionName}.apk"
    }

    doLast {
        println("✅ APK copied to distribution folder")
        println("📁 Location: ../distribution/f3set-tennis-analyzer-v${android.defaultConfig.versionName}.apk")
    }
}

tasks.register("prepareDistribution") {
    dependsOn("printVersionInfo", "assembleRelease", "copyReleaseApk")

    doLast {
        println("")
        println("🎉 Distribution package ready!")
        println("📁 Check the ../distribution/ folder")
        println("")
        println("🚀 Next steps:")
        println("   1. Test the APK on Android devices")
        println("   2. Upload to your GitHub Pages repository")
        println("   3. Update your website with new version info")
    }
}