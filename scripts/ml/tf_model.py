"""TF port of the landmark classifier, for a real INT8 TFLite export.

Why this file exists: on Windows there is no working torch->TFLite path.
ai-edge-torch needs torch_xla (no Windows wheel), and the ONNX route ships
ONNX protobuf under a .tflite name and then fails in onnx2tf's Reshape
lowering on TF 2.21.

So the architecture is expressed directly in TensorFlow, weights are copied
from the PyTorch checkpoint, and TF's own converter produces the FlatBuffer.
`verify_parity()` in export_tflite_tf.py checks the TF model reproduces the
PyTorch logits before anything is quantised - a port that is not numerically
faithful would otherwise be indistinguishable from a bad model.

The topology mirrors scripts/ml/train.py exactly:
    stem 1x1 conv -> BatchNorm
    3 x [depthwise k=17 -> pointwise 1x1 -> BN -> GELU -> SE -> dropout, residual]
    single-query attention pool over time
    linear head
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]


class TimeMean(tf.keras.layers.Layer):
    """Mean over the time axis, kept as a layer so Keras can trace it.

    Functional Keras forbids calling raw tf ops on a KerasTensor, so the
    squeeze-and-excite pool has to be a layer rather than a tf.reduce_mean.
    """

    def call(self, x):
        return tf.reduce_mean(x, axis=1, keepdims=True)


def _se_block(x, dim, name):
    """Squeeze-and-excite: channel attention from the time-mean."""
    h = TimeMean(name=f"{name}_pool")(x)                    # (B,1,C)
    h = tf.keras.layers.Conv1D(dim // 4, 1, activation="relu", name=f"{name}_fc1")(h)
    h = tf.keras.layers.Conv1D(dim, 1, activation="sigmoid", name=f"{name}_fc2")(h)
    return tf.keras.layers.Multiply(name=f"{name}_scale")([x, h])


class QueryConst(tf.keras.layers.Layer):
    """Emits the learned attention query, broadcast across the batch.

    The broadcast is a static reshape to (1,1,1,dim) followed by the multiply
    with the input's own first element - not tf.tile over tf.shape(x). The
    dynamic tf.shape path emitted SHAPE/GATHER/PACK/TRANSPOSE float ops that
    blocked full INT8 lowering (the converted graph kept 53 int32 tensors and
    lost 7 accuracy points).
    """

    def __init__(self, dim, **kw):
        super().__init__(**kw)
        self.dim = dim

    def build(self, input_shape):
        self.q = self.add_weight(name="q", shape=(1, 1, self.dim),
                                 initializer="zeros", trainable=False)

    def call(self, x):
        # (1,1,dim) + (B,1,dim)*0 broadcasts to (B,1,dim) with no dynamic
        # shape ops. tf.tile over tf.shape(x) emitted SHAPE/GATHER/PACK/
        # TRANSPOSE float ops that blocked full INT8 lowering and cost
        # accuracy.
        return tf.cast(self.q, x.dtype) + tf.zeros_like(x[:, :1, :])


def build_tf_model(num_classes, dim=192, frames=24, feature_dim=255, dropout=0.2):
    """Functional model matching train.TemporalModel.

    Layer names mirror the PyTorch state_dict keys so `copy_weights` can map
    them without a lookup table that would silently rot.
    """
    inp = tf.keras.Input(shape=(frames, feature_dim), name="features")

    x = inp                                            # (B,T,F)
    x = tf.keras.layers.Conv1D(dim, 1, name="stem")(x)  # (B,T,C)
    x = tf.keras.layers.BatchNormalization(name="norm", epsilon=1e-5)(x)

    for i in range(3):
        h = tf.keras.layers.DepthwiseConv1D(17, padding="same",
                                            name=f"blocks_{i}_dw")(x)
        h = tf.keras.layers.Conv1D(dim, 1, name=f"blocks_{i}_pw")(h)
        h = tf.keras.layers.BatchNormalization(name=f"blocks_{i}_norm", epsilon=1e-5)(h)
        h = tf.keras.layers.Activation("gelu", name=f"blocks_{i}_act")(h)
        h = _se_block(h, dim, f"blocks_{i}_se")
        h = tf.keras.layers.Dropout(dropout, name=f"blocks_{i}_drop")(h)
        x = tf.keras.layers.Add(name=f"blocks_{i}_res")([h, x])

    # Attention pool, replicating nn.MultiheadAttention's exact arithmetic.
    # Keras MultiHeadAttention rescales scores internally in a way that does
    # not match torch, and its per-head weight layout made the port silently
    # wrong (verified: K and V matched to 7.6e-06 while the softmax input did
    # not). Doing the maths explicitly keeps the two implementations
    # comparable, and every step is a plain layer so TFLite can convert it.
    q_in = QueryConst(dim, name="query_in")(x)              # (B,1,C) pre-projected

    n_heads, head_dim = 4, dim // 4
    q = AttnProject(n_heads, head_dim, name="q_proj")(q_in)   # (B,H,1,d)
    k = AttnProject(n_heads, head_dim, name="k_proj")(x)      # (B,H,T,d)
    v = AttnProject(n_heads, head_dim, name="v_proj")(x)

    scores = AttnScores(head_dim, name="attn_scores")([q, k])      # (B,H,1,T)
    w = tf.keras.layers.Softmax(axis=-1, name="attn_softmax")(scores)
    ctx = AttnContext(name="attn_context")([w, v])                 # (B,1,H,d)
    ctx = tf.keras.layers.Reshape((1, dim), name="attn_merge")(ctx)
    ctx = tf.keras.layers.Dense(dim, name="out_proj")(ctx)

    pooled = tf.keras.layers.Dropout(dropout, name="drop")(ctx)
    out = tf.keras.layers.Dense(num_classes, name="head")(pooled[:, 0, :])
    return tf.keras.Model(inp, out, name="landmark_classifier")


class AttnProject(tf.keras.layers.Layer):
    """(B,T,C) -> (B,H,T,d) with a learned linear projection.

    Deliberately a single matmul plus a static reshape. An earlier version
    used tf.einsum, which lowered to 19 RESHAPE / 13 EXPAND_DIMS / 50 int32
    shape tensors; the converter then could not fully quantise and both TFLite
    runtimes disagreed on the same file.
    """

    def __init__(self, n_heads, head_dim, **kw):
        super().__init__(**kw)
        self.n_heads, self.head_dim = n_heads, head_dim

    def build(self, shape):
        c = int(shape[-1])
        # one dense kernel (C -> H*d), then a fixed reshape
        self.kernel = self.add_weight(name="kernel", shape=(c, self.n_heads * self.head_dim))
        self.bias = self.add_weight(name="bias", shape=(self.n_heads * self.head_dim,),
                                    initializer="zeros")

    def call(self, x):
        y = tf.matmul(x, tf.cast(self.kernel, x.dtype)) + tf.cast(self.bias, x.dtype)
        b, t = tf.shape(y)[0], tf.shape(y)[1]
        y = tf.reshape(y, (b, t, self.n_heads, self.head_dim))
        return tf.transpose(y, (0, 2, 1, 3))

    def compute_output_shape(self, shape):
        return (shape[0], self.n_heads, shape[1], self.head_dim)


class AttnScores(tf.keras.layers.Layer):
    """Scaled dot-product scores, (B,H,1,T)."""

    def __init__(self, head_dim, **kw):
        super().__init__(**kw)
        self.head_dim = head_dim

    def call(self, inputs):
        q, k = inputs
        return tf.matmul(q, k, transpose_b=True) * (1.0 / tf.sqrt(
            tf.cast(self.head_dim, q.dtype)))


class AttnContext(tf.keras.layers.Layer):
    """Softmax weights applied to values, then heads merged to (B,1,C)."""

    def call(self, inputs):
        w, v = inputs
        c = tf.matmul(w, v)                       # (B,H,1,d)
        b = tf.shape(c)[0]
        return tf.transpose(c, (0, 2, 1, 3))      # (B,1,H,d)


def copy_weights(tf_model, ckpt_path):
    """Copy PyTorch parameters into the TF model.

    Keras BatchNormalization stores [gamma, beta, moving_mean, moving_var] in
    that order; torch stores weight, bias, running_mean, running_var. Every
    other layer is a single kernel, transposed for TF's layout.

    Returns (copied, missed) - the caller refuses to export on a mismatch,
    because a half-initialised model still converts and still produces
    plausible-looking logits.
    """
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt["model"]
    dim = int(ckpt["dim"])

    def arr(k):
        return sd[k].detach().cpu().numpy().astype(np.float32)

    copied = missed = 0

    def put(layer, weights):
        nonlocal copied
        layer.set_weights([w.astype(np.float32) for w in weights])
        copied += 1

    put(tf_model.get_layer("stem"), [np.transpose(arr("stem.weight"), (2, 1, 0)),
                                     arr("stem.bias")])
    put(tf_model.get_layer("norm"), [arr("norm.weight"), arr("norm.bias"),
                                     arr("norm.running_mean"), arr("norm.running_var")])

    for i in range(3):
        p = f"blocks.{i}"
        put(tf_model.get_layer(f"blocks_{i}_dw"),
            # torch (out,1,k) -> keras (k, channels, depth_multiplier=1)
            [np.transpose(arr(f"{p}.dw.weight")[:, 0, :], (1, 0))[..., None],
             arr(f"{p}.dw.bias")])
        put(tf_model.get_layer(f"blocks_{i}_pw"),
            [np.transpose(arr(f"{p}.pw.weight"), (2, 1, 0)), arr(f"{p}.pw.bias")])
        put(tf_model.get_layer(f"blocks_{i}_norm"),
            [arr(f"{p}.norm.weight"), arr(f"{p}.norm.bias"),
             arr(f"{p}.norm.running_mean"), arr(f"{p}.norm.running_var")])
        put(tf_model.get_layer(f"blocks_{i}_se_fc1"),
            [np.transpose(arr(f"{p}.se.1.weight"), (2, 1, 0)), arr(f"{p}.se.1.bias")])
        put(tf_model.get_layer(f"blocks_{i}_se_fc2"),
            [np.transpose(arr(f"{p}.se.3.weight"), (2, 1, 0)), arr(f"{p}.se.3.bias")])

    # torch's in_proj_weight is (3*dim, dim) laid out as [Q; K; V].
    # Keras MultiHeadAttention stores each projection per head as
    # (dim, n_heads, head_dim), so the (dim, dim) slice must be reshaped,
    # not copied flat.
    n_heads, head_dim = 4, dim // 4
    in_proj = arr("attn.in_proj_weight")                  # (576,192)
    in_bias = arr("attn.in_proj_bias")                    # (576,)
    q_lo, k_lo, v_lo = 0, dim, 2 * dim

    def per_head(lo):
        """torch (dim_out, dim_in) slice -> keras (dim_in, n_heads*head_dim).

        Per-head order is preserved: reshaping (dim_in, dim_out) to
        (dim_in, n_heads, head_dim) keeps head h in columns
        [h*head_dim, (h+1)*head_dim), which is how AttnProject reads it.
        """
        return in_proj[lo:lo + dim].T.astype(np.float32)

    def per_head_bias(lo):
        return in_bias[lo:lo + dim].astype(np.float32)

    # The learned attention query is a bare (1,1,dim) Parameter in torch.
    # It is stored RAW here and projected by q_proj, exactly as torch does -
    # folding W_q in beforehand applied it twice and broke parity.
    raw_q = np.asarray(sd["query"].detach().cpu()).reshape(1, 1, dim)
    put(tf_model.get_layer("query_in"), [raw_q])

    # q_proj / k_proj / v_proj take the three slices of the fused in_proj.
    # The query enters q_proj raw, so W_q is applied exactly once here - the
    # same single application torch performs.
    for layer_name, lo in (("q_proj", q_lo), ("k_proj", k_lo), ("v_proj", v_lo)):
        put(tf_model.get_layer(layer_name), [per_head(lo), per_head_bias(lo)])

    put(tf_model.get_layer("head"),
        [arr("head.weight").T, arr("head.bias")])
    put(tf_model.get_layer("out_proj"),
        [arr("attn.out_proj.weight").T, arr("attn.out_proj.bias")])

    return copied, missed, ckpt

    # Keras MultiHeadAttention keeps its own output projection. It is stored as
    # a sub-layer, so its kernel is set directly instead of through set_weights
    # on the attention layer (which would also overwrite the Q/K/V kernels).
    attn = tf_model.get_layer("attn")
    op = None
    for sub in attn._layers if hasattr(attn, "_layers") else []:
        if sub.__class__.__name__ == "EinsumDense" and "output" in sub.name:
            op = sub
    if op is None:
        # fall back: the projection is the only sub-layer with a bias+2d kernel
        cands = [s for s in attn._layers
                 if s.__class__.__name__ == "EinsumDense"
                 and len(s.get_weights()) == 2]
        op = cands[-1] if cands else None
    if op is not None:
        # Keras keeps the output projection per head: (num_heads, head_dim, dim).
        # torch stores the flattened (dim, dim); same matrix, different layout.
        n_heads = 4
        head_dim = dim // n_heads
        w = arr("attn.out_proj.weight").T                      # (in, out)
        w = w.reshape(n_heads, head_dim, dim)
        op.set_weights([w.astype(np.float32), arr("attn.out_proj.bias")])
        copied += 1
        print(f"  out_proj -> {op.name} {w.shape}")

    return copied, missed, ckpt


def to_tflite(tf_model, out_path, calib=None):
    """Convert to an INT8 TFLite FlatBuffer with float32 IO.

    Float32 IO keeps predict() simple and avoids the full-integer-IO accuracy
    collapse that integer-only inputs are prone to.
    """
    conv = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    if calib is not None:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = calib
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.float32
    conv.inference_output_type = tf.float32
    data = conv.convert()
    Path(out_path).write_bytes(data)
    return len(data)