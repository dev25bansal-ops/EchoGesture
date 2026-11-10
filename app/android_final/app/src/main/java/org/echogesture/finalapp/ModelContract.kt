package org.echogesture.finalapp

import android.content.res.AssetManager
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/**
 * Pinned identity of the frozen EchoGesture landmark candidate. Nothing here
 * substitutes a different artifact: the classifier is FP16 weights with
 * float32 IO (not INT8), and both binary hashes are checked before use.
 */
data class ModelIdentity(
    val fileName: String,
    val sha256: String,
    val frames: Int,
    val featureDim: Int,
    val classes: Int,
    val quantization: String
) {
    val shortSha: String get() = sha256.take(12)
    val displayName: String
        get() = "$fileName · $quantization · $classes classes"
}

data class LoadedModel(
    val identity: ModelIdentity,
    val classifierModel: ByteBuffer,
    val labels: List<String>
)

object ModelContract {
    const val CLASSIFIER_ASSET = "landmark_classifier_fp16.tflite"
    const val DETECTOR_ASSET = "hand_landmarker.task"
    const val LABELS_ASSET = "landmark_labels.json"
    const val META_ASSET = "landmark_model_meta.json"

    const val EXPECTED_FRAMES = 24
    const val EXPECTED_FEATURE_DIM = 255
    const val EXPECTED_CLASSES = 100

    const val CLASSIFIER_SHA256 =
        "a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4"
    const val DETECTOR_SHA256 =
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"

    /**
     * Loads and validates the four bundled assets. Throws [IllegalStateException]
     * with an actionable message on any drift from the frozen contract.
     */
    fun load(assets: AssetManager): LoadedModel {
        val classifierBytes = readAsset(assets, CLASSIFIER_ASSET)
        val classifierHash = sha256(classifierBytes)
        check(classifierHash == CLASSIFIER_SHA256) {
            "Classifier hash mismatch: ${classifierHash.take(12)}… != ${
                CLASSIFIER_SHA256.take(12)
            }…"
        }

        val detectorBytes = readAsset(assets, DETECTOR_ASSET)
        val detectorHash = sha256(detectorBytes)
        check(detectorHash == DETECTOR_SHA256) {
            "Detector hash mismatch: ${detectorHash.take(12)}… != ${
                DETECTOR_SHA256.take(12)
            }…"
        }

        val meta = JSONObject(String(readAsset(assets, META_ASSET), Charsets.UTF_8))
        val frames = meta.getInt("frames")
        val featureDim = meta.getInt("feature_dim")
        val classes = meta.getInt("classes")
        val quantization = meta.optString("quantization", "unknown")

        check(frames == EXPECTED_FRAMES) { "Metadata frames $frames != $EXPECTED_FRAMES" }
        check(featureDim == EXPECTED_FEATURE_DIM) {
            "Metadata feature_dim $featureDim != $EXPECTED_FEATURE_DIM"
        }
        check(featureDim == FeatureTransform.FEATURE_DIM) {
            "Feature implementation produces ${FeatureTransform.FEATURE_DIM}, metadata says $featureDim"
        }
        check(classes == EXPECTED_CLASSES) { "Metadata classes $classes != $EXPECTED_CLASSES" }

        val labels = parseLabels(String(readAsset(assets, LABELS_ASSET), Charsets.UTF_8))
        check(labels.size == classes) { "Labels ${labels.size} != model classes $classes" }

        val model = ByteBuffer.allocateDirect(classifierBytes.size)
            .order(ByteOrder.nativeOrder())
            .apply { put(classifierBytes); rewind() }

        return LoadedModel(
            identity = ModelIdentity(
                fileName = CLASSIFIER_ASSET,
                sha256 = classifierHash,
                frames = frames,
                featureDim = featureDim,
                classes = classes,
                quantization = if (quantization.isEmpty()) "unknown" else quantization
            ),
            classifierModel = model,
            labels = labels
        )
    }

    /** landmark_labels.json is a {label: index} object; index order is the class axis. */
    fun parseLabels(json: String): List<String> {
        val root = JSONObject(json)
        val keys = root.keys()
        val labels = arrayOfNulls<String>(root.length())
        check(labels.isNotEmpty()) { "Vocabulary is empty" }
        while (keys.hasNext()) {
            val label = keys.next()
            val index = root.getInt(label)
            check(index in labels.indices) { "Label '$label' has out-of-range index $index" }
            check(labels[index] == null) { "Duplicate label index $index" }
            labels[index] = label.uppercase()
        }
        return labels.mapIndexed { index, label ->
            checkNotNull(label) { "Vocabulary index $index is missing" }
        }
    }

    private fun readAsset(assets: AssetManager, name: String): ByteArray =
        assets.open(name).use { it.readBytes() }

    fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256")
            .digest(bytes)
            .joinToString("") { "%02x".format(it) }
}