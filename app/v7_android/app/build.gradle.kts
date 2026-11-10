// V7 Android slice — pinned versions (do not float).
// CameraX 1.3.4 (stable, API 24+), MediaPipe Tasks Vision 0.10.x,
// TensorFlow Lite 2.16.x (stable, matches host ai-edge-litert semantics).
plugins {
    id("com.android.application") version "8.5.2"
    id("org.jetbrains.kotlin.android") version "2.0.20"
}

android {
    namespace = "org.echogesture.v7"
    compileSdk = 35
    defaultConfig {
        applicationId = "org.echogesture.v7"
        minSdk = 24
        targetSdk = 35
    }
    buildTypes {
        release {
            isMinifyEnabled = false // keep slice debuggable; enable + shrink for submission APK
        }
    }
}

dependencies {
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("com.google.mediapipe:tasks-vision:0.10.14")
    implementation("androidx.camera:camera-camera2:1.3.4")
    implementation("androidx.camera:camera-lifecycle:1.3.4")
    implementation("androidx.camera:camera-view:1.3.4")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
}
