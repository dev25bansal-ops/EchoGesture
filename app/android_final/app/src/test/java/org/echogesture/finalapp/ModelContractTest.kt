package org.echogesture.finalapp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * JVM tests for the REAL [ModelContract] validation.
 *
 * Two groups, deliberately separated:
 *
 *  - **Semantic tests** need no model bytes. They exercise the production
 *    parsing and validation logic against text fixtures, so they run on every
 *    checkout, including one that has none of the restricted weights.
 *
 *  - **Weight tests** need the full, verified asset set. When that set is
 *    genuinely absent they report as skipped with an explicit reason. When it
 *    is present but INCOMPLETE or CORRUPT they FAIL — an asset set that is
 *    half there is a broken build, not an unavailable one.
 */
class ModelContractTest {

    private fun repoRoot(): File {
        // Gradle runs unit tests with the module directory as CWD:
        // app/android_final/app -> repo root is four levels up.
        var dir = File(".").absoluteFile
        repeat(6) {
            if (File(dir, "models").isDirectory) return dir
            dir = dir.parentFile ?: return@repeat
        }
        error("could not locate repo root from ${File(".").absolutePath}")
    }

    private fun assetDirs(): List<File> {
        val root = repoRoot()
        return listOf(File(root, "app/android_final/app/src/main/assets"), File(root, "models"))
    }

    private fun source(dir: File, overrides: Map<String, ByteArray> = emptyMap()) =
        ModelContract.AssetSource { name -> overrides[name] ?: File(dir, name).readBytes() }

    private fun readOrThrow(dir: File, absent: String) =
        ModelContract.AssetSource { n ->
            if (n == absent) throw java.io.FileNotFoundException("$n (absent)")
            File(dir, n).readBytes()
        }

    // ---- asset-set classification ------------------------------------------

    private enum class Assets { COMPLETE, ABSENT, BROKEN }

    /**
     * Classifies the on-disk asset set.
     *
     *  - COMPLETE: every pinned file is present and hashes to its pin.
     *  - ABSENT: no restricted binary is present at all. This is the normal
     *    state of a public checkout: `models/` carries the labels and the
     *    sidecar, which are tracked, but not the weights.
     *  - BROKEN: some but not all binaries are present, or a present file
     *    hashes wrong. A half-supplied or corrupt asset set must fail rather
     *    than quietly behave like a checkout that never had one.
     */
    private fun classify(): Pair<Assets, String> {
        val binaries = (ModelContract.MEMBER_SHA256.keys + ModelContract.DETECTOR_ASSET)
        val sidecars = listOf(ModelContract.LABELS_ASSET, ModelContract.META_ASSET)
        val required = binaries + sidecars

        val candidates = assetDirs().map { dir ->
            dir to required.filter { File(dir, it).isFile }
        }
        val (dir, found) = candidates.maxByOrNull { it.second.size }!!

        if (found.isEmpty()) {
            return Assets.ABSENT to "no model assets present in any candidate directory"
        }
        val binariesFound = found.filter { it in binaries }
        if (binariesFound.isEmpty()) {
            // Tracked text assets only, which is exactly what the repository
            // commits. The weights live outside Git by design.
            return Assets.ABSENT to
                "$dir has the tracked labels/sidecar but none of the restricted weights"
        }
        if (found.size != required.size) {
            return Assets.BROKEN to
                "asset set at $dir is incomplete; missing ${required - found.toSet()}"
        }
        val bad = required.filter { name ->
            val expected = when (name) {
                in ModelContract.MEMBER_SHA256 -> ModelContract.MEMBER_SHA256[name]
                ModelContract.DETECTOR_ASSET -> ModelContract.DETECTOR_SHA256
                else -> null
            } ?: return@filter false
            ModelContract.sha256(File(dir, name).readBytes()) != expected
        }
        if (bad.isNotEmpty()) return Assets.BROKEN to "asset set at $dir has hash mismatch in $bad"
        return Assets.COMPLETE to dir.absolutePath
    }

    /** Returns the asset dir when COMPLETE; skips with a reason when ABSENT; fails when BROKEN. */
    /**
     * A synthetic sidecar. The real one carries `members_detail`; these tests
     * vary the member list, so the detail block is derived from the same list
     * to keep each case about the thing it is testing.
     */
    private fun sidecar(members: List<String>): ByteArray {
        val detail = members.joinToString(",", "[", "]") { m ->
            val z = m.contains("_z_")
            """{"file":"$m","use_z":$z}"""
        }
        return """
            {"frames":24,"feature_dim":255,"classes":100,"quantization":"float16",
             "members":${members.joinToString(",", "[", "]") { "\"$it\"" }},
             "members_detail":$detail}
        """.trimIndent().toByteArray()
    }

    private fun requireCompleteAssets(): File {
        val (state, detail) = classify()
        when (state) {
            Assets.COMPLETE -> return File(detail)
            Assets.BROKEN -> throw AssertionError(
                "Model assets are present but unusable: $detail. A partial or corrupt " +
                    "asset set is a build failure, not a reason to skip."
            )
            Assets.ABSENT -> {
                println(
                    "[ModelContractTest] weight-dependent tests SKIPPED: $detail. " +
                        "The semantic tests in this class and the source-only vocabulary " +
                        "test still ran."
                )
                org.junit.Assume.assumeTrue(
                    "model weights are not published; $detail", false
                )
                error("unreachable")
            }
        }
    }

    // ---- A. semantic tests: no weights needed ------------------------------

    @Test
    fun absentMembersListIsRejectedNotTreatedAsSingleModel() {
        val meta = """{"frames":24,"feature_dim":255,"classes":100,"quantization":"float16"}"""
        val err = runCatching { ModelContract.parseMembers(org.json.JSONObject(meta)) }
            .exceptionOrNull()
        assertTrue("a meta without 'members' must fail", err is IllegalStateException)
        assertTrue(
            "message must say a single-model asset is unsupported: ${err?.message}",
            err!!.message!!.contains("single-model")
        )
    }

    @Test
    fun pathTraversalAndDuplicatesAreRejected() {
        fun errFor(membersJson: String): Throwable? = runCatching {
            ModelContract.parseMembers(org.json.JSONObject("""{"members":$membersJson}"""))
        }.exceptionOrNull()

        for (bad in listOf(
            """["../escape.tflite"]""", """["dir/x.tflite"]""",
            """["..\\escape.tflite"]""", """["a.tflite","a.tflite"]""",
            """[""]""", "[]"
        )) {
            assertTrue("expected rejection for $bad", errFor(bad) != null)
        }
        assertEquals(
            listOf("a.tflite", "b.tflite"),
            ModelContract.parseMembers(
                org.json.JSONObject("""{"members":["a.tflite","b.tflite"]}""")
            )
        )
    }

    @Test
    fun memberOrderIsPreservedNotSetCompared() {
        // Order is semantic: aggregation is a mean in this order and the
        // combined hash concatenates in it. parseMembers must not sort or
        // deduplicate through a set.
        val two = org.json.JSONObject(
            """{"members":["landmark_ens_rob71_fp16.tflite","landmark_ens_tcn21_fp16.tflite"]}"""
        )
        assertEquals(
            listOf("landmark_ens_rob71_fp16.tflite", "landmark_ens_tcn21_fp16.tflite"),
            ModelContract.parseMembers(two)
        )
    }

    @Test
    fun shippedMetadataDeclaresThePinnedContract() {
        // The tracked sidecar, read as data. This is what the app will load, so
        // its three shape numbers, quantization and ordered members are checked
        // here without touching a single weight.
        val meta = org.json.JSONObject(
            File(repoRoot(), "models/${ModelContract.META_ASSET}").readText(Charsets.UTF_8)
        )
        assertEquals(ModelContract.EXPECTED_FRAMES, meta.getInt("frames"))
        assertEquals(ModelContract.EXPECTED_FEATURE_DIM, meta.getInt("feature_dim"))
        assertEquals(ModelContract.EXPECTED_CLASSES, meta.getInt("classes"))
        assertEquals(ModelContract.EXPECTED_QUANTIZATION, meta.getString("quantization"))
        // Every member must declare its feature width; a member with no entry
        // has an indeterminate width and the loader refuses it.
        assertEquals(
            ModelContract.parseMembers(meta).toSet(),
            ModelContract.parseMembersDetail(meta).keys.toSet()
        )
        assertEquals(
            ModelContract.MEMBER_SHA256.keys.toList(),
            ModelContract.parseMembers(meta)
        )
        // Order is load-bearing: aggregation is a mean in this order and the
        // combined hash concatenates in it.
        assertEquals(9, ModelContract.parseMembers(meta).size)
    }

    @Test
    fun vocabularyParsesToAContiguousOrderedClassAxis() {
        val labels = ModelContract.parseLabels(
            File(repoRoot(), "models/${ModelContract.LABELS_ASSET}").readText(Charsets.UTF_8)
        )
        assertEquals(ModelContract.EXPECTED_CLASSES, labels.size)
        assertEquals("WHATFOR1", labels[8])
        assertTrue("labels must be unique", labels.toSet().size == labels.size)
    }

    @Test
    fun malformedVocabularyIsRejected() {
        for (bad in listOf(
            """{"A":0,"B":0}""",     // duplicate index
            """{"A":0,"B":2}""",     // gap, so index 1 is missing
            """{"A":-1}""",          // negative index
            """{}"""                 // empty
        )) {
            val err = runCatching { ModelContract.parseLabels(bad) }.exceptionOrNull()
            assertTrue("vocabulary $bad must be rejected, got $err", err is IllegalStateException)
        }
    }

    // ---- B. weight tests: need the complete verified set -------------------

    @Test
    fun realAssetsLoadAllNineMembersInMetaOrder() {
        val dir = requireCompleteAssets()
        val loaded = ModelContract.load(source(dir))
        assertEquals(ModelContract.MEMBER_SHA256.size, loaded.members.size)
        assertEquals(9, loaded.members.size)
        // Four x,y (255-wide) then five x,y,z (381-wide), per members_detail.
        assertEquals(4, loaded.members.count { !it.useZ })
        assertEquals(5, loaded.members.count { it.useZ })
        assertEquals(ModelContract.MEMBER_SHA256.keys.toList(), loaded.identity.memberNames)
        assertEquals(ModelContract.ENSEMBLE_SHA256, loaded.identity.combinedSha256)
        assertEquals(100, loaded.labels.size)
        assertEquals("WHATFOR1", loaded.labels[8])
        loaded.members.forEach { assertEquals(ModelContract.MEMBER_SHA256[it.name], it.sha256) }
    }

    @Test
    fun aMissingListedMemberIsAHardError() {
        val dir = requireCompleteAssets()
        val absent = "landmark_ens_rob71_fp16.tflite"
        val err = runCatching {
            ModelContract.load(readOrThrow(dir, absent = absent))
        }.exceptionOrNull()
        assertTrue("a missing listed member must fail, got $err", err != null)
        assertTrue(
            "failure must name the missing member, got: ${err?.message}",
            err!!.message!!.contains(absent)
        )
    }

    @Test
    fun corruptMemberBytesAreRejectedWithHashMismatch() {
        val dir = requireCompleteAssets()
        val target = "landmark_ens_tcnm62_fp16.tflite"
        val bytes = File(dir, target).readBytes()
        bytes[bytes.size / 2] = (bytes[bytes.size / 2] + 1).toByte()
        val err = runCatching {
            ModelContract.load(source(dir, mapOf(target to bytes)))
        }.exceptionOrNull()
        assertTrue("corrupt member must fail", err is IllegalStateException)
        assertTrue(
            "must report a hash mismatch for $target, got: ${err?.message}",
            err!!.message!!.contains("hash mismatch") && err.message!!.contains(target)
        )
    }

    @Test
    fun threeMemberListFailsTheCombinedIdentity() {
        val dir = requireCompleteAssets()
        val meta = sidecar(
            listOf(
                "landmark_ens_tcn21_fp16.tflite",
                "landmark_ens_tcnm61_fp16.tflite",
                "landmark_ens_tcnm62_fp16.tflite"
            )
        )
        val err = runCatching {
            ModelContract.load(source(dir, mapOf(ModelContract.META_ASSET to meta)))
        }.exceptionOrNull()
        assertTrue("3-of-4 members must fail the combined-hash check", err is IllegalStateException)
        assertTrue(
            "must be a combined-hash failure, got: ${err?.message}",
            err!!.message!!.contains("combined hash")
        )
    }

    @Test
    fun reorderedMembersFailTheCombinedIdentity() {
        val dir = requireCompleteAssets()
        val reordered = ModelContract.MEMBER_SHA256.keys.toList().reversed()
        val meta = sidecar(reordered)
        val err = runCatching {
            ModelContract.load(source(dir, mapOf(ModelContract.META_ASSET to meta)))
        }.exceptionOrNull()
        assertTrue("a reordered member list must fail", err is IllegalStateException)
        assertTrue(
            "must be a combined-hash failure, got: ${err?.message}",
            err!!.message!!.contains("combined hash")
        )
    }

    @Test
    fun unpinnedMemberIsRejectedEvenIfBytesAreValid() {
        val dir = requireCompleteAssets()
        val meta = sidecar(
            listOf(
                "landmark_ens_tcn21_fp16.tflite",
                "landmark_ens_tcnm61_fp16.tflite",
                "landmark_ens_tcnm62_fp16.tflite",
                "landmark_ens_rob71_fp16.tflite",
                "landmark_ens_unknown_fp16.tflite"
            )
        )
        val err = runCatching {
            ModelContract.load(source(dir, mapOf(ModelContract.META_ASSET to meta)))
        }.exceptionOrNull()
        assertTrue("an unpinned member must fail", err is IllegalStateException)
        assertTrue(
            "must refuse an unverified binary: ${err?.message}",
            err!!.message!!.contains("not pinned")
        )
    }

    @Test
    fun wrongFrameCountFeatureDimOrClassCountIsRejected() {
        val dir = requireCompleteAssets()
        val good = File(dir, ModelContract.META_ASSET).readText(Charsets.UTF_8)
        for ((field, value) in listOf(
            "frames" to 32, "feature_dim" to 225, "classes" to 99
        )) {
            val mutated = good.replace(
                Regex("\"$field\"\\s*:\\s*\\d+"), "\"$field\": $value"
            )
            assertNotEquals("mutation must change the file", good, mutated)
            val err = runCatching {
                ModelContract.load(
                    source(dir, mapOf(ModelContract.META_ASSET to mutated.toByteArray()))
                )
            }.exceptionOrNull()
            assertTrue("wrong $field=$value must fail, got $err", err is IllegalStateException)
            assertTrue(
                "message must name $field: ${err?.message}",
                err!!.message!!.contains(field)
            )
        }
    }

    @Test
    fun int8QuantizationIsRejected() {
        val dir = requireCompleteAssets()
        val good = File(dir, ModelContract.META_ASSET).readText(Charsets.UTF_8)
        val mutated = good.replace(
            Regex("\"quantization\"\\s*:\\s*\"[^\"]*\""), "\"quantization\": \"int8\""
        )
        assertNotEquals("mutation must change the file", good, mutated)
        val err = runCatching {
            ModelContract.load(
                source(dir, mapOf(ModelContract.META_ASSET to mutated.toByteArray()))
            )
        }.exceptionOrNull()
        assertTrue("INT8 must be refused", err is IllegalStateException)
        assertTrue(
            "message must name quantization: ${err?.message}",
            err!!.message!!.contains("quantization")
        )
    }

    @Test
    fun corruptLabelMappingIsRejected() {
        val dir = requireCompleteAssets()
        val bad = """{"A":0,"B":0}""".toByteArray()
        val err = runCatching {
            ModelContract.load(source(dir, mapOf(ModelContract.LABELS_ASSET to bad)))
        }.exceptionOrNull()
        assertTrue("a corrupted label mapping must fail", err is IllegalStateException)
    }

    @Test
    fun extraProvenanceFieldsDoNotChangeWhatLoads() {
        val dir = requireCompleteAssets()
        val good = File(dir, ModelContract.META_ASSET).readText(Charsets.UTF_8)
        val withProvenance = good.trimEnd().removeSuffix("}") +
            ",\n  \"v4_3_provenance\": \"added by test\",\n" +
            "  \"evaluated_on\": \"val_30\",\n" +
            "  \"notes\": \"provenance only\"\n}\n"
        val plain = ModelContract.load(source(dir))
        val annotated = ModelContract.load(
            source(dir, mapOf(ModelContract.META_ASSET to withProvenance.toByteArray()))
        )
        assertEquals(plain.identity, annotated.identity)
        assertEquals(plain.members.map { it.name }, annotated.members.map { it.name })
        assertEquals(plain.labels, annotated.labels)
    }

    @Test
    fun buildTimeHashesMatchTheRuntimePins() {
        val dir = requireCompleteAssets()
        val loaded = ModelContract.load(source(dir))
        loaded.members.forEach { m ->
            assertEquals("pinned hash drift for ${m.name}", ModelContract.MEMBER_SHA256[m.name], m.sha256)
            assertNotEquals("", m.sha256)
        }
        assertEquals(
            ModelContract.DETECTOR_SHA256,
            ModelContract.sha256(File(dir, ModelContract.DETECTOR_ASSET).readBytes())
        )
    }
}