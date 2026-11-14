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
    val sha256: String,
    /** True when this member's features are x,y,z (381 wide) per the sidecar. */
    val useZ: Boolean = false
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
    /** Width of the x,y members. z members are 381; the sidecar says which. */
    const val EXPECTED_FEATURE_DIM = 255
    const val EXPECTED_CLASSES = 100
    /** The shipped members are float16 weights with float32 IO; not INT8. */
    const val EXPECTED_QUANTIZATION = "float16"

    /**
     * Pinned per-member SHA-256. A member present in the meta list but absent
     * here is rejected: an unpinned binary is an unverified binary.
     */
    val MEMBER_SHA256: Map<String, String> = mapOf(
        // Four x,y (255-wide) members, re-exported WITH the attention mask on
        // 2026-11-14. Same filenames as the V4.2 bundle, different binaries:
        // e.g. landmark_ens_tcn21_fp16.tflite was 112279bc…, and is now
        // af031d89…. Filenames are therefore not an identity; these hashes are.
        "landmark_ens_tcn21_fp16.tflite" to
            "af031d898c98c97abcf66ae7538166316faac458b7b938fe9e3390fce31e5d4c",
        "landmark_ens_tcnm61_fp16.tflite" to
            "b0a97aa45c6797b4b85e2f6a1b03945881a94dad5519cb71be47e474fdea54f0",
        "landmark_ens_tcnm62_fp16.tflite" to
            "1d247397d57e982196e68f3f230adeeabbfcdf673ba01ed541eadd54d874ea8d",
        "landmark_ens_rob71_fp16.tflite" to
            "105218a89fde1d107441422254805dfb38cc7df25a216ff986801bc15dccb89a",
        // Five x,y,z (381-wide) members.
        "landmark_ens_mirror_handdrop_tcn_z_s0_fp16.tflite" to
            "387749f81673719bd4a24813f33c6880c6117ed7be9d6bd50853250f8082901e",
        "landmark_ens_mirror_handdrop_tcn_z_s1_fp16.tflite" to
            "02a2dc70fbd769f793c8f8915c8a30cc568d901bf2e35bffe1752aa762000a35",
        "landmark_ens_mirror_handdrop_tcn_z_s2_fp16.tflite" to
            "3b97a2d416faec002bb84a6d2e47c131bb97941f074feed248a2c5a99c1c3b3d",
        "landmark_ens_mirror_handdrop_tcn_z_s7_fp16.tflite" to
            "2f13d0eba1f34e68f3e3ac1b2c645d9f75d1953918d69be6d58ca00a1f2b36b2",
        "landmark_ens_mirror_handdrop_tcn_z_s13_fp16.tflite" to
            "3d3a61e08d6a4d2a0a8064ae3b63692a55b55a2004ba37a428308a22b6336fba"
    )

    /** SHA-256 of the nine members concatenated in meta order. */
    const val ENSEMBLE_SHA256 =
        "995cc8d0b6f6ac25abb9f7ddca01c3ce71edf8bc2d34e34f4813a2b99dd3b4bb"

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
        // The sidecar must declare a use_z flag for every member: the width a
        // member expects is not inferable from its filename, and guessing it
        // would feed a 381-wide model a 255-wide tensor.
        val detailUseZ = parseMembersDetail(meta)
        check(classes == EXPECTED_CLASSES) { "Metadata classes $classes != $EXPECTED_CLASSES" }

        val memberNames = parseMembers(meta)
        val members = memberNames.map { name ->
            val pinned = MEMBER_SHA256[name]
                ?: throw IllegalStateException(
                    "Ensemble member '$name' from $META_ASSET is not pinned in " +
                        "ModelContract.MEMBER_SHA256; refusing to load an unverified binary."
                )
            val useZ = detailUseZ[name]
                ?: throw IllegalStateException(
                    "Member '$name' has no entry in $META_ASSET members_detail; " +
                        "its feature width would have to be guessed."
                )
            val bytes = source.read(name)
            val hash = sha256(bytes)
            check(hash == pinned) {
                "Member '$name' hash mismatch: ${hash.take(12)}… != ${pinned.take(12)}…"
            }
            MemberAsset(name = name, bytes = bytes, sha256 = hash, useZ = useZ)
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
     * `members_detail` from the sidecar: member basename -> use_z.
     *
     * Every member must appear. A member with no entry has an indeterminate
     * feature width, and inferring it from the filename is exactly the kind of
     * guess that feeds a 381-wide model a 255-wide tensor.
     */
    fun parseMembersDetail(meta: JSONObject): Map<String, Boolean> {
        check(meta.has("members_detail")) {
            "$META_ASSET has no 'members_detail'; per-member feature width is unknown"
        }
        val array = meta.getJSONArray("members_detail")
        val out = HashMap<String, Boolean>(array.length())
        for (i in 0 until array.length()) {
            val entry = array.getJSONObject(i)
            val file = entry.optString("file", "")
            check(file.isNotEmpty()) { "$META_ASSET members_detail[$i] has no 'file'" }
            out[file] = entry.optBoolean("use_z", false)
        }
        return out
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