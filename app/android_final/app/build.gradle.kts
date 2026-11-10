plugins {
    id("com.android.application") version "8.5.2"
    id("org.jetbrains.kotlin.android") version "2.0.20"
}

// Copy the four named assets from the root models/ directory as documented in
// README.md. Binary hashes are pinned so a classifier cannot be substituted.
val modelAssets = mapOf(
    "hand_landmarker.task" to
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    "landmark_classifier_fp16.tflite" to
        "a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4",
    "landmark_labels.json" to null,
    "landmark_model_meta.json" to null
)

modelAssets.forEach { (name, expectedHash) ->
    val asset = layout.projectDirectory.file("src/main/assets/$name").asFile
    if (!asset.isFile) {
        throw GradleException(
            "Missing model asset ${asset.absolutePath}. Copy it from the root " +
                "models/ directory using the README.md prepare-assets commands."
        )
    }
    if (expectedHash != null) {
        val digest = java.security.MessageDigest.getInstance("SHA-256")
        asset.inputStream().buffered().use { input ->
            val buffer = ByteArray(64 * 1024)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        val actual = digest.digest().joinToString("") { "%02x".format(it) }
        if (actual != expectedHash) {
            throw GradleException("$name SHA-256 is $actual; expected $expectedHash")
        }
    }
}

android {
    namespace = "org.echogesture.finalapp"
    compileSdk = 34

    defaultConfig {
        applicationId = "org.echogesture.finalapp"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.camera:camera-camera2:1.3.4")
    implementation("androidx.camera:camera-lifecycle:1.3.4")
    implementation("androidx.camera:camera-view:1.3.4")
    implementation("com.google.mediapipe:tasks-vision:0.10.14")
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    testImplementation("junit:junit:4.13.2")
}
