import java.security.MessageDigest

plugins {
    id("com.android.application") version "8.5.2"
    id("org.jetbrains.kotlin.android") version "2.0.20"
}

// Copy the named assets from the root models/ directory as documented in
// README.md. Binary hashes are pinned so a classifier cannot be substituted.
//
// The member ORDER is pinned too, not just the set: aggregation is a mean over
// the members in this order, and the combined hash is the concatenation in this
// order. A reordering produces a different ensemble, so `members` is compared
// as an ordered list further down.
val expectedMembers = listOf(
    "landmark_ens_tcn21_fp16.tflite",
    "landmark_ens_tcnm61_fp16.tflite",
    "landmark_ens_tcnm62_fp16.tflite",
    "landmark_ens_rob71_fp16.tflite"
)
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
// Every weight and the detector are byte-pinned: a swapped classifier cannot
// get past this, and those bytes are what decide what the app runs.
//
// Text assets are pinned with line endings normalized to LF. The pins were
// taken on a Linux checkout; on Windows the same tracked file arrives with
// CRLF and hashes differently, which failed the build over a checkout setting
// rather than over a wrong model.
//
// landmark_model_meta.json is deliberately NOT byte-pinned. It is a sidecar
// whose meaning is the ordered member list, the three shape numbers and the
// quantization, and every one of those is validated semantically below.
// Pinning its bytes failed the build for cosmetic reasons (line endings, and
// provenance keys this repo's copy does not carry). Extra keys that nothing
// reads at runtime — evaluation metrics, for instance — must not change what
// the app runs, so they are allowed and ignored.
val textAssets = mapOf(
    "landmark_labels.json" to
        "e78f5028e97e2eaf86aff65af403af65061eebef010f0fbd41012293e8019a4a"
)
val binaryAssets = mapOf(
    "hand_landmarker.task" to
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
) + memberHashes
val modelAssets = textAssets + binaryAssets

/** JSON text assets are compared with line endings normalized to LF. */
fun sha256Of(file: java.io.File, normalizeNewlines: Boolean = false): String {
    val digest = MessageDigest.getInstance("SHA-256")
    val bytes = file.readBytes()
    val payload = if (normalizeNewlines) {
        bytes.toString(Charsets.UTF_8).replace("\r\n", "\n").toByteArray(Charsets.UTF_8)
    } else {
        bytes
    }
    return digest.digest(payload).joinToString("") { "%02x".format(it) }
}

val assetsDir = layout.projectDirectory.dir("src/main/assets")

fun sha256Of(name: String, normalizeNewlines: Boolean = name in textAssets): String =
    sha256Of(assetsDir.file(name).asFile, normalizeNewlines)

/**
 * The complete set of pinned assets, digest-verified. This is the ONLY thing
 * that may run before an APK is packaged; source-only tasks call
 * [validateMetadata] alone so they do not need the restricted weights.
 */
fun requirePinnedAssets() {
    modelAssets.forEach { (name, expectedHash) ->
        val asset = assetsDir.file(name).asFile
        if (!asset.isFile) {
            throw GradleException(
                "Missing model asset ${asset.absolutePath}. Copy it from the root " +
                    "models/ directory using the README.md prepare-assets commands; " +
                    "all ${memberHashes.size} ensemble members and the detector " +
                    "are required to package the app."
            )
        }
        val actual = sha256Of(name)
        if (actual != expectedHash) {
            throw GradleException("$name SHA-256 is $actual; expected $expectedHash")
        }
    }
}

/**
 * Semantic validation of landmark_model_meta.json against the pinned contract.
 * Needs only the sidecar (a tiny text file), never the weights, so it can run
 * on a source-only checkout.
 *
 * Optional provenance keys are allowed and ignored. Everything the runtime
 * actually reads is checked here, because a sidecar that disagrees with the
 * pinned binaries would otherwise package cleanly and then refuse to load.
 */
fun validateMetadata(metaFile: java.io.File) {
    val meta = try {
        @Suppress("UNCHECKED_CAST")
        groovy.json.JsonSlurper().parse(metaFile) as? Map<String, Any?>
            ?: throw GradleException("${metaFile.name} must contain a JSON object")
    } catch (e: org.codehaus.groovy.runtime.typehandling.GroovyCastException) {
        throw GradleException("${metaFile.name} must contain a JSON object, not an array")
    }

    fun intField(key: String, expected: Int) {
        val raw = meta[key]
        if (raw !is Number || raw.toInt() != raw.toDouble().toInt()) {
            throw GradleException(
                "${metaFile.name} '$key' must be the integer $expected, got '$raw'"
            )
        }
        if (raw.toInt() != expected) {
            throw GradleException(
                "${metaFile.name} '$key' is ${raw.toInt()}; expected $expected"
            )
        }
    }
    intField("frames", 24)
    intField("feature_dim", 255)
    intField("classes", 100)

    val quantization = meta["quantization"]
    if (quantization?.toString()?.lowercase() != "float16") {
        throw GradleException(
            "${metaFile.name} declares quantization='$quantization'; the shipped "
                + "ensemble is FP16 and every published metric was measured on it. "
                + "An INT8 member would emit different logits, so it must not ship here."
        )
    }

    @Suppress("UNCHECKED_CAST")
    val declared = (meta["members"] as? List<*>)
        ?.map { it as? String ?: throw GradleException(
            "${metaFile.name} 'members' entries must be strings, got '$it'") }
        .orEmpty()
    if (declared.isEmpty()) {
        throw GradleException(
            "${metaFile.name} declares no members; the build cannot verify a "
                + "single-classifier fallback, which this app no longer supports."
        )
    }
    declared.forEach { name ->
        if (name.isEmpty() || name != name.substringAfterLast('/').substringAfterLast('\\')
            || ".." in name
        ) {
            throw GradleException(
                "${metaFile.name} has unsafe member name '$name' (must be a plain basename)"
            )
        }
    }
    if (declared.size != declared.toSet().size) {
        throw GradleException(
            "${metaFile.name} lists a member more than once: $declared. Averaging the "
                + "same member twice is not the pinned ensemble."
        )
    }
    if (declared.size != expectedMembers.size) {
        throw GradleException(
            "${metaFile.name} declares ${declared.size} members, expected "
                + "${expectedMembers.size}: $declared"
        )
    }
    if (declared != expectedMembers) {
        val reordered = declared.toSet() == expectedMembers.toSet()
        val message = if (reordered) {
            "${metaFile.name} member order $declared does not match the pinned " +
                "order $expectedMembers. Aggregation is a mean in this order " +
                "and the combined hash concatenates in it."
        } else {
            "${metaFile.name} members $declared do not match the pinned set " +
                "$expectedMembers; update expectedMembers, memberHashes and the " +
                "metadata together."
        }
        throw GradleException(message)
    }
}

// Validate the sidecar at configuration time when it is present, so an edit to
// it fails fast even on a checkout that has no weights. When it is absent the
// build still configures: a source-only checkout must be able to compile and
// run the JVM unit tests, which do not package an APK. The packaging gate
// below requires the sidecar, so no APK can be produced without it.
assetsDir.file("landmark_model_meta.json").asFile
    .takeIf { it.isFile }
    ?.let { validateMetadata(it) }

// The binary/detector gate is attached to the tasks that actually produce or
// install the packaged app, so a source-only checkout can still compile and run
// the JVM unit tests. Any task that packages the APK depends on this one.
val verifyPinnedAssets by tasks.registering {
    group = "verification"
    description = "Verifies the metadata sidecar and every pinned model weight and detector."
    doLast {
        val metaFile = assetsDir.file("landmark_model_meta.json").asFile
        if (!metaFile.isFile) {
            throw GradleException(
                "Missing ${metaFile.absolutePath}. Copy it from the root models/ "
                    + "directory using the README.md prepare-assets commands; the "
                    + "runtime reads the member list from it."
            )
        }
        validateMetadata(metaFile)
        requirePinnedAssets()
    }
}

tasks.matching {
    it.name in setOf(
        "assembleDebug", "assembleRelease", "assembleAndroidTest",
        "packageDebug", "packageRelease",
        "connectedDebugAndroidTest"
    ) || it.name.startsWith("package") && it.name.endsWith("Bundle")
}.configureEach {
    dependsOn(verifyPinnedAssets)
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
    implementation("com.google.mediapipe:tasks-vision:1.2.3")
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    testImplementation("junit:junit:4.13.2")
    // org.json is provided by the Android framework at runtime; the JVM unit
    // tests need a real implementation. Test-scoped only, never in the APK.
    testImplementation("org.json:json:20240303")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
}
