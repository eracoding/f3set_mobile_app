# F3Set Tennis Analyzer - ProGuard Rules
# Updated to fix R8 errors with PyTorch dependencies

# Keep application class
-keep public class com.example.aivideoanalyzer.AIVideoAnalyzerApplication

# Keep all activities and fragments
-keep public class * extends android.app.Activity
-keep public class * extends androidx.fragment.app.Fragment
-keep public class * extends androidx.appcompat.app.AppCompatActivity

# ===== PYTORCH DEPENDENCIES - CRITICAL FIXES =====
# Keep all PyTorch classes and prevent R8 optimization issues
-keep class org.pytorch.** { *; }
-dontwarn org.pytorch.**

# Keep Facebook JNI classes (used by PyTorch)
-keep class com.facebook.jni.** { *; }
-dontwarn com.facebook.jni.**

# Keep PyTorch Lite classes
-keep class org.pytorch.lite.** { *; }
-dontwarn org.pytorch.lite.**

# Keep TorchVision classes
-keep class org.pytorch.torchvision.** { *; }
-dontwarn org.pytorch.torchvision.**

# ===== MISSING ANNOTATIONS FIX =====
# Fix for missing javax.annotation classes
-dontwarn javax.annotation.**
-dontwarn javax.annotation.Nullable
-dontwarn javax.annotation.Nonnull
-dontwarn javax.annotation.concurrent.**

# Add missing annotation classes
-keep class javax.annotation.** { *; }

# ===== KOTLIN AND COROUTINES =====
# Keep Kotlin coroutines
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}
-dontwarn kotlinx.coroutines.flow.**
-dontwarn kotlinx.coroutines.**

# Keep Kotlin metadata
-keep class kotlin.Metadata { *; }
-dontwarn kotlin.**

# ===== APP-SPECIFIC CLASSES =====
# Keep data classes and models
-keep class com.example.aivideoanalyzer.domain.model.** { *; }
-keep class com.example.aivideoanalyzer.data.** { *; }
-keep class com.example.aivideoanalyzer.ml.** { *; }

# Keep ViewModels
-keep class * extends androidx.lifecycle.ViewModel { *; }
-keep class * extends androidx.lifecycle.AndroidViewModel { *; }

# ===== EXPORT FUNCTIONALITY =====
# Keep OpenCSV classes for export functionality
-keep class com.opencsv.** { *; }
-keep class au.com.bytecode.opencsv.** { *; }
-dontwarn com.opencsv.**

# ===== NAVIGATION COMPONENT =====
# Keep Navigation component classes
-keep class androidx.navigation.** { *; }
-dontwarn androidx.navigation.**

# ===== VIEW BINDING =====
# Keep ViewBinding classes
-keep class * implements androidx.viewbinding.ViewBinding {
    public static *** inflate(android.view.LayoutInflater);
    public static *** inflate(android.view.LayoutInflater, android.view.ViewGroup, boolean);
    public static *** bind(android.view.View);
}

# ===== MATERIAL DESIGN =====
# Keep Material Design components
-keep class com.google.android.material.** { *; }
-dontwarn com.google.android.material.**

# ===== ANDROIDX LIBRARIES =====
# Keep AndroidX classes
-keep class androidx.** { *; }
-dontwarn androidx.**

# ===== ENUM CLASSES =====
# Keep enum classes
-keepclassmembers enum * {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# ===== SERIALIZABLE CLASSES =====
# Keep serializable classes
-keepclassmembers class * implements java.io.Serializable {
    static final long serialVersionUID;
    private static final java.io.ObjectStreamField[] serialPersistentFields;
    private void writeObject(java.io.ObjectOutputStream);
    private void readObject(java.io.ObjectInputStream);
    java.lang.Object writeReplace();
    java.lang.Object readResolve();
}

# ===== REFLECTION =====
# Keep classes that use reflection
-keepclassmembers class * {
    @androidx.annotation.Keep *;
}

# ===== REMOVE LOGGING IN RELEASE =====
# Remove logging in release builds (optional - comment out if you want logs)
-assumenosideeffects class android.util.Log {
    public static boolean isLoggable(java.lang.String, int);
    public static int v(...);
    public static int i(...);
    public static int w(...);
    public static int d(...);
    public static int e(...);
}

# ===== OPTIMIZATION SETTINGS =====
# R8 optimization settings
-allowaccessmodification
-dontpreverify

# Keep line numbers for crash reports
-keepattributes SourceFile,LineNumberTable,Signature,*Annotation*
-renamesourcefileattribute SourceFile

# ===== ADDITIONAL MISSING CLASSES FIXES =====
# Fix for common missing classes in Android projects
-dontwarn java.lang.instrument.**
-dontwarn sun.misc.Unsafe
-dontwarn sun.nio.ch.**

# Fix for OkHttp (if used indirectly by dependencies)
-dontwarn okhttp3.**
-dontwarn okio.**

# Fix for Gson (if used by dependencies)
-dontwarn com.google.gson.**

# Fix for Jackson (if used by dependencies)
-dontwarn com.fasterxml.jackson.**

# ===== PYTORCH NATIVE LIBRARIES =====
# Keep native methods for PyTorch
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep classes with native methods (fixed syntax)
-keepclasseswithmembers class * {
    native <methods>;
}

# ===== FINAL SAFETY NETS =====
# Don't warn about missing classes that are handled at runtime
-ignorewarnings

# Alternative: Use this if you want to see warnings but not fail the build
# -dontwarn **