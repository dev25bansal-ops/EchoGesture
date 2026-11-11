package org.echogesture.finalapp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * JVM tests for the REAL [ModelContract] asset/identity validation.
 *
 * These call the production validation code ([ModelContract.load] over an
 * [ModelContract.AssetSource]); they are not a re-implementation. The model
 * binaries are not redistributable and are absent from a clean checkout, so the
 * whole class skips unless `models/landmark_model_meta.json` is present. The
 * fixtures used here are the actual assets under
 * `app/src/main/assets/` when present, else the root `models/` directory.
 */
class ModelContractTest {

    private fun repoRoot(): File {
        // Gradle runs unit tests with the module directory as CWD:
        // app/android_final/app -> repo root is four levels up.
        var dir = File(".").absoluteFile
        repeat(6) {
            if (File(dir, "models/landmark_model_meta.json").isFile) return dir
            dir = dir.parentFile ?: return@repeat
        }
        error("could not locate repo root from ${File(".").absolutePath}")
    }

    private fun assetDirs(): List<File> {
        val root = repoRoot()
        return listOf(File(root, "app/android_final/app/src/main/assets"), File(root, "models"))
            .filter { File(it, ModelContract.META_ASSET).isFile }
    }

    private fun source(dir: File, overrides: Map<String, ByteArray> = emptyMap()) =
        ModelContract.AssetSource { name -> overrides[name] ?: File(dir, name).readBytes() }

    private fun readOrThrow(dir: File, absent: String) =
        ModelContract.AssetSource { n ->
            if (n == absent) throw java.io.FileNotFoundException("$n (absent)")
            File(dir, n).readBytes()
        }

    private fun requireAssets(): File {
        val dirs = assetDirs()
        assumeTrue(
            "model weights are not published; skipping real-bytes contract test",
            dirs.isNotEmpty()
        )
        return dirs.first()
    }

    @Test
    fun realAssetsLoadAllFourMembersInMetaOrder() {
        val dir = requireAssets()
        val loaded = ModelContract.load(source(dir))
        assertEquals(4, loaded.members.size)
        assertEquals(
            listOf(
                "landmark_ens_tcn21_fp16.tflite",
                "landmark_ens_tcnm61_fp16.tflite",
                "landmark_ens_tcnm62_fp16.tflite",
                "landmark_ens_rob71_fp16.tflite"
            ),
            loaded.identity.memberNames
        )
        assertEquals(ModelContract.ENSEMBLE_SHA256, loaded.identity.combinedSha256)
        assertEquals(100, loaded.labels.size)
        assertEquals("WHATFOR1", loaded.labels[8])
        // Each member carries the pinned hash it was verified against.
        loaded.members.forEach { assertEquals(ModelContract.MEMBER_SHA256[it.name], it.sha256) }
    }

    @Test
    fun aMissingListedMemberIsAHardError() {
        val dir = requireAssets()
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
        val dir = requireAssets()
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
    fun metadataWithoutMembersIsRejectedNotTreatedAsSingleModel() {
        val dir = requireAssets()
        val meta = """
            {"frames":24,"feature_dim":255,"classes":100,"quantization":"float16"}
        """.trimIndent().toByteArray()
        val err = runCatching {
            ModelContract.load(source(dir, mapOf(ModelContract.META_ASSET to meta)))
        }.exceptionOrNull()
        assertTrue("meta without 'members' must fail", err is IllegalStateException)
        assertTrue(
            "message must say a single-model asset is unsupported: ${err?.message}",
            err!!.message!!.contains("single-model")
        )
    }

    @Test
    fun droppingAMemberFromMetadataChangesThePinnedIdentity() {
        val dir = requireAssets()
        val meta = """
            {"frames":24,"feature_dim":255,"classes":100,"quantization":"float16",
             "members":["landmark_ens_tcn21_fp16.tflite",
                        "landmark_ens_tcnm61_fp16.tflite",
                        "landmark_ens_tcnm62_fp16.tflite"]}
        """.trimIndent().toByteArray()
        // A 3-member list parses, but its combined hash cannot match the pinned
        // 4-member hash: dropping a member must fail the identity check.
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
    fun unpinnedMemberIsRejectedEvenIfBytesAreValid() {
        val dir = requireAssets()
        val meta = """
            {"frames":24,"feature_dim":255,"classes":100,"quantization":"float16",
             "members":["landmark_ens_tcn21_fp16.tflite",
                        "landmark_ens_tcnm61_fp16.tflite",
                        "landmark_ens_tcnm62_fp16.tflite",
                        "landmark_ens_rob71_fp16.tflite",
                        "landmark_ens_unknown_fp16.tflite"]}
        """.trimIndent().toByteArray()
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
    fun pathTraversalAndDuplicatesAreRejected() {
        fun errFor(membersJson: String): Throwable? = runCatching {
            ModelContract.parseMembers(org.json.JSONObject("""{"members":$membersJson}"""))
        }.exceptionOrNull()

        for (bad in listOf("""["../escape.tflite"]""", """["dir/x.tflite"]""",
                           """["..\\escape.tflite"]""", """["a.tflite","a.tflite"]""",
                           """[""]""", "[]")) {
            assertTrue("expected rejection for $bad", errFor(bad) != null)
        }
        assertEquals(
            listOf("a.tflite", "b.tflite"),
            ModelContract.parseMembers(org.json.JSONObject("""{"members":["a.tflite","b.tflite"]}"""))
        )
    }

    @Test
    fun buildTimeHashesMatchTheRuntimePins() {
        // The Gradle script pins the same values; this asserts the two copies
        // agree so a build that validates cannot ship a runtime that rejects.
        val dir = requireAssets()
        val loaded = ModelContract.load(source(dir))
        loaded.members.forEach { m ->
            assertEquals("pinned hash drift for ${m.name}", ModelContract.MEMBER_SHA256[m.name], m.sha256)
            assertNotEquals("", m.sha256)
        }
        assertEquals(ModelContract.DETECTOR_SHA256, ModelContract.sha256(File(dir, ModelContract.DETECTOR_ASSET).readBytes()))
    }
}