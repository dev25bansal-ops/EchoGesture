import java.security.MessageDigest

plugins {
    id("com.android.application") version "8.5.2"
    id("org.jetbrains.kotlin.android") version "2.0.20"
}

// Copy the named assets from the root models/ directory as documented in
// README.md. Binary hashes are pinned so a classifier cannot be substituted.
// Every ensemble member listed in landmark_model_meta.json is required.
val memberHashes = mapOf(
    "landmark_ens_tcn21_fp16.tflite" to
        "112279bc86090470b82b20b510746a946220322c25ec84cb3706a538c73dc22a",
    "landmark_ens_tcnm61_fp16.tflite" to
        "01a261d8eb6c2c4d8b501f0ce1a36db338365d17fddabc63528a7941b1390f63",
    "landmark_ens_tcnm62_fp16.tflite" to
        "27058a85e032bee3af444fb1538ff0bf32fd38fa2b01ce01a2f25c3a428d20b3",
    "landmark_ens_rob71_fp16.tflite" to
        "08417e72da59150ac9adf749e6324c30fb001d4a401394e32f125d3276d1ab21"
)
val modelAssets = mapOf(
    "hand_landmarker.task" to
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    "landmark_labels.json" to null,
    "landmark_model_meta.json" to null
) + memberHashes

modelAssets.forEach { (name, expectedHash) ->
    val asset = layout.projectDirectory.file("src/main/assets/$name").asFile
    if (!asset.isFile) {
        throw GradleException(
            "Missing model asset ${asset.absolutePath}. Copy it from the root " +
                "models/ directory using the README.md prepare-assets commands; " +
                "all ${memberHashes.size} ensemble members are required."
        )
    }
    if (expectedHash != null) {
        val digest = MessageDigest.getInstance("SHA-256")
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
    // org.json is provided by the Android framework at runtime; the JVM unit
    // tests need a real implementation. Test-scoped only, never in the APK.
    testImplementation("org.json:json:20240303")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
}
