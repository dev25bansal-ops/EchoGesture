package org.echogesture.finalapp

import android.content.res.AssetManager
import org.json.JSONObject
import java.security.MessageDigest

/**
 * Pinned identity of the frozen EchoGesture landmark ensemble. Nothing here
 * substitutes a different artifact: every member is FP16 weights with float32
 * IO (not INT8), and each binary hash is checked before use.
 */
data class ModelIdentity(
    /** Human label, e.g. "4-model ensemble". */
    val fileName: String,
    /** Ordered member basenames exactly as listed in the meta sidecar. */
    val memberNames: List<String>,
    val frames: Int,
    val featureDim: Int,
    val classes: Int,
    val quantization: String,
    /** SHA-256 of the ordered concatenation of all member bytes. */
    val combinedSha256: String
) {
    val shortSha: String get() = combinedSha256.take(12)
    val displayName: String
        get() = "${memberNames.size}-model ensemble · $quantization · $classes classes"
}

/** One ensemble member's raw bytes plus its own pinned hash. */
data class MemberAsset(
    val name: String,
    val bytes: ByteArray,
    val sha256: String
)

data class LoadedModel(
    val identity: ModelIdentity,
    /** In meta order; exactly the list in landmark_model_meta.json. */
    val members: List<MemberAsset>,
    val labels: List<String>
)

object ModelContract {
    const val DETECTOR_ASSET = "hand_landmarker.task"
    const val LABELS_ASSET = "landmark_labels.json"
    const val META_ASSET = "landmark_model_meta.json"

    /** Reads named assets' bytes. Production uses [AssetManager]; tests use files. */
    fun interface AssetSource {
        fun read(name: String): ByteArray
    }

    const val EXPECTED_FRAMES = 24
    const val EXPECTED_FEATURE_DIM = 255
    const val EXPECTED_CLASSES = 100
    /** The shipped members are float16 weights with float32 IO; not INT8. */
    const val EXPECTED_QUANTIZATION = "float16"

    /**
     * Pinned per-member SHA-256. A member present in the meta list but absent
     * here is rejected: an unpinned binary is an unverified binary.
     */
    val MEMBER_SHA256: Map<String, String> = mapOf(
        "landmark_ens_tcn21_fp16.tflite" to
            "112279bc86090470b82b20b510746a946220322c25ec84cb3706a538c73dc22a",
        "landmark_ens_tcnm61_fp16.tflite" to
            "01a261d8eb6c2c4d8b501f0ce1a36db338365d17fddabc63528a7941b1390f63",
        "landmark_ens_tcnm62_fp16.tflite" to
            "27058a85e032bee3af444fb1538ff0bf32fd38fa2b01ce01a2f25c3a428d20b3",
        "landmark_ens_rob71_fp16.tflite" to
            "08417e72da59150ac9adf749e6324c30fb001d4a401394e32f125d3276d1ab21"
    )

    /** SHA-256 of the four members concatenated in meta order. */
    const val ENSEMBLE_SHA256 =
        "0e4c392732e7da7e77632620dc72e4afce8b1a5aa00234dbf7a13a1049a7ad93"

    const val DETECTOR_SHA256 =
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"

    /**
     * Loads and validates the detector, meta, labels and EVERY ensemble member
     * listed in the meta sidecar. Throws [IllegalStateException] with an
     * actionable message on any drift from the frozen contract.
     *
     * An incomplete asset set (a listed member missing from assets, or a member
     * absent from the pinned hash table) is a hard error — this never silently
     * runs a smaller ensemble.
     */
    fun load(assets: AssetManager): LoadedModel = load(AssetSource { name -> readAsset(assets, name) })

    /**
     * The same validation over any byte reader. [AssetManager] is Android-only,
     * so this entry point lets the JVM tests exercise the real contract against
     * the real bytes instead of a re-implementation. Production always calls
     * [load] with the Android [AssetManager].
     */
    fun load(source: AssetSource): LoadedModel {
        val meta = JSONObject(String(source.read(META_ASSET), Charsets.UTF_8))
        val frames = meta.getInt("frames")
        val featureDim = meta.getInt("feature_dim")
        val classes = meta.getInt("classes")
        val quantization = meta.optString("quantization", "unknown")
        // The shipped members are FP16. An INT8 or otherwise re-quantized member
        // would still load and still aggregate, but its logits would not be the
        // ones any published metric was measured on, so refuse it rather than
        // silently serving a differently-quantized ensemble.
        check(quantization.equals(EXPECTED_QUANTIZATION, ignoreCase = true)) {
            "Metadata quantization '$quantization' != expected '$EXPECTED_QUANTIZATION'"
        }
        check(frames == EXPECTED_FRAMES) { "Metadata frames $frames != $EXPECTED_FRAMES" }
        check(featureDim == EXPECTED_FEATURE_DIM) {
            "Metadata feature_dim $featureDim != $EXPECTED_FEATURE_DIM"
        }
        check(featureDim == FeatureTransform.FEATURE_DIM) {
            "Feature implementation produces ${FeatureTransform.FEATURE_DIM}, metadata says $featureDim"
        }
        check(classes == EXPECTED_CLASSES) { "Metadata classes $classes != $EXPECTED_CLASSES" }

        val memberNames = parseMembers(meta)
        val members = memberNames.map { name ->
            val pinned = MEMBER_SHA256[name]
                ?: throw IllegalStateException(
                    "Ensemble member '$name' from $META_ASSET is not pinned in " +
                        "ModelContract.MEMBER_SHA256; refusing to load an unverified binary."
                )
            val bytes = source.read(name)
            val hash = sha256(bytes)
            check(hash == pinned) {
                "Member '$name' hash mismatch: ${hash.take(12)}… != ${pinned.take(12)}…"
            }
            MemberAsset(name = name, bytes = bytes, sha256 = hash)
        }
        check(members.size == memberNames.size) {
            "Ensemble member set was not fully loaded (${members.size}/${memberNames.size})"
        }

        val detectorBytes = source.read(DETECTOR_ASSET)
        val detectorHash = sha256(detectorBytes)
        check(detectorHash == DETECTOR_SHA256) {
            "Detector hash mismatch: ${detectorHash.take(12)}… != ${DETECTOR_SHA256.take(12)}…"
        }

        val labels = parseLabels(String(source.read(LABELS_ASSET), Charsets.UTF_8))
        check(labels.size == classes) { "Labels ${labels.size} != model classes $classes" }

        // Hash the ordered concatenation without copying it into one array.
        val combined = sha256Of(members.map { it.bytes })
        check(combined == ENSEMBLE_SHA256) {
            "Ensemble combined hash ${combined.take(12)}… != ${ENSEMBLE_SHA256.take(12)}…"
        }

        return LoadedModel(
            identity = ModelIdentity(
                fileName = "landmark ensemble",
                memberNames = memberNames,
                frames = frames,
                featureDim = featureDim,
                classes = classes,
                quantization = if (quantization.isEmpty()) "unknown" else quantization,
                combinedSha256 = combined
            ),
            members = members,
            labels = labels
        )
    }

    /**
     * Ordered member basenames from the meta sidecar. Missing/empty/unsafe
     * entries are rejected rather than tolerated — a dropped member would
     * change predictions without any visible failure.
     *
     * Note: an unsafe name fails [parseMembers]; a basename that is simply
     * absent from the assets fails later, in [readAsset]. Both are load errors.
     */
    fun parseMembers(meta: JSONObject): List<String> {
        check(meta.has("members")) {
            "$META_ASSET has no 'members' list; a single-model asset is not a " +
                "supported ensemble configuration."
        }
        val array = meta.getJSONArray("members")
        check(array.length() > 0) { "$META_ASSET 'members' must be a non-empty list" }
        val names = ArrayList<String>(array.length())
        for (i in 0 until array.length()) {
            val name = array.optString(i, "")
            check(name.isNotEmpty()) { "$META_ASSET 'members[$i]' is not a non-empty string" }
            check(name == name.substringAfterLast('/').substringAfterLast('\\')) {
                "Unsafe ensemble member name '$name' (must be a plain basename)"
            }
            check(".." !in name) { "Unsafe ensemble member name '$name'" }
            check(name !in names) { "Duplicate ensemble member '$name'" }
            names.add(name)
        }
        return names
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

    /** SHA-256 of the parts in order, fed streaming (no concatenated copy). */
    fun sha256Of(parts: List<ByteArray>): String {
        val digest = MessageDigest.getInstance("SHA-256")
        parts.forEach(digest::update)
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}