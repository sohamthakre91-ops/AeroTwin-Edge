"""
AeroTwin Edge
Convert trained sklearn MLP into a Qualcomm AI Hub-friendly pure ONNX graph.

Architecture:
    14 inputs
       ↓
    StandardScaler
       ↓
    Dense 32 + ReLU
       ↓
    Dense 16 + ReLU
       ↓
    Dense 7
       ↓
    Softmax
       ↓
    7 fault probabilities

The graph intentionally uses only standard ONNX operators:
    Sub
    Div
    MatMul
    Add
    Relu
    Softmax

This avoids:
    - ai.onnx.ml operators
    - sklearn runtime operators
    - string outputs
    - sequence/map probability outputs
    - dynamic input shapes
"""

from pathlib import Path

import joblib
import numpy as np
import onnx

from onnx import helper, numpy_helper, TensorProto


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "aerotwin_edge_mlp.joblib"
ONNX_PATH = ROOT / "models" / "aerotwin_edge_mlp.onnx"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("AeroTwin Edge -> Qualcomm AI Hub ONNX Converter")
print("=" * 70)


# ============================================================
# LOAD SKLEARN PIPELINE
# ============================================================

print("\nLoading trained model...")
print(f"Model: {MODEL_PATH}")

pipeline = joblib.load(MODEL_PATH)

print("Model loaded successfully.")

if not hasattr(pipeline, "named_steps"):
    raise ValueError("Expected a sklearn Pipeline.")


# ============================================================
# EXTRACT SCALER + MLP
# ============================================================

scaler = pipeline.named_steps["scaler"]
mlp = pipeline.named_steps["classifier"]


# ============================================================
# MODEL INFORMATION
# ============================================================

feature_count = scaler.n_features_in_
classes = mlp.classes_

print("\nModel information:")
print(f"Input features : {feature_count}")
print(f"Output classes : {len(classes)}")
print(f"Classes        : {list(classes)}")
print(f"Hidden layers  : {mlp.hidden_layer_sizes}")


# ============================================================
# SAFETY CHECKS
# ============================================================

if feature_count != 14:
    raise ValueError(
        f"Expected 14 input features, but found {feature_count}."
    )

if tuple(mlp.hidden_layer_sizes) != (32, 16):
    raise ValueError(
        f"Expected architecture (32, 16), "
        f"but found {mlp.hidden_layer_sizes}."
    )

if len(classes) != 7:
    raise ValueError(
        f"Expected 7 classes, but found {len(classes)}."
    )


# ============================================================
# EXTRACT TRAINED PARAMETERS
# ============================================================

mean = np.asarray(
    scaler.mean_,
    dtype=np.float32,
)

scale = np.asarray(
    scaler.scale_,
    dtype=np.float32,
)

weights = [
    np.asarray(weight, dtype=np.float32)
    for weight in mlp.coefs_
]

biases = [
    np.asarray(bias, dtype=np.float32)
    for bias in mlp.intercepts_
]


print("\nExtracted parameters:")

print(f"Scaler mean : {mean.shape}")
print(f"Scaler scale: {scale.shape}")

for i, (weight, bias) in enumerate(zip(weights, biases)):
    print(
        f"Layer {i}: "
        f"weights={weight.shape}, "
        f"bias={bias.shape}"
    )


# ============================================================
# CREATE ONNX GRAPH CONTAINERS
# ============================================================

nodes = []
initializers = []


# ============================================================
# INPUT
# ============================================================

input_name = "float_input"

input_tensor = helper.make_tensor_value_info(
    input_name,
    TensorProto.FLOAT,
    [1, feature_count],
)


# ============================================================
# OUTPUT
# ============================================================

output_name = "probabilities"

output_tensor = helper.make_tensor_value_info(
    output_name,
    TensorProto.FLOAT,
    [1, len(classes)],
)


# ============================================================
# STANDARD SCALER
#
# scaled = (input - mean) / scale
# ============================================================

mean_name = "scaler_mean"
scale_name = "scaler_scale"

initializers.append(
    numpy_helper.from_array(
        mean,
        name=mean_name,
    )
)

initializers.append(
    numpy_helper.from_array(
        scale,
        name=scale_name,
    )
)


# ------------------------------------------------------------
# SUB
# ------------------------------------------------------------

nodes.append(
    helper.make_node(
        "Sub",
        inputs=[
            input_name,
            mean_name,
        ],
        outputs=[
            "centered",
        ],
        name="Scaler_Sub",
    )
)


# ------------------------------------------------------------
# DIV
# ------------------------------------------------------------

nodes.append(
    helper.make_node(
        "Div",
        inputs=[
            "centered",
            scale_name,
        ],
        outputs=[
            "scaled",
        ],
        name="Scaler_Div",
    )
)


current = "scaled"


# ============================================================
# MLP
# ============================================================

for i, (weight, bias) in enumerate(zip(weights, biases)):

    weight_name = f"weight_{i}"
    bias_name = f"bias_{i}"

    matmul_output = f"matmul_{i}"
    add_output = f"dense_{i}"

    # --------------------------------------------------------
    # WEIGHTS
    # --------------------------------------------------------

    initializers.append(
        numpy_helper.from_array(
            weight,
            name=weight_name,
        )
    )

    # --------------------------------------------------------
    # BIAS
    # --------------------------------------------------------

    initializers.append(
        numpy_helper.from_array(
            bias,
            name=bias_name,
        )
    )

    # --------------------------------------------------------
    # MATMUL
    # --------------------------------------------------------

    nodes.append(
        helper.make_node(
            "MatMul",
            inputs=[
                current,
                weight_name,
            ],
            outputs=[
                matmul_output,
            ],
            name=f"MLP_MatMul_{i}",
        )
    )

    # --------------------------------------------------------
    # ADD BIAS
    # --------------------------------------------------------

    nodes.append(
        helper.make_node(
            "Add",
            inputs=[
                matmul_output,
                bias_name,
            ],
            outputs=[
                add_output,
            ],
            name=f"MLP_Add_{i}",
        )
    )

    current = add_output

    # --------------------------------------------------------
    # RELU
    #
    # Only hidden layers use ReLU.
    # Final layer goes directly into Softmax.
    # --------------------------------------------------------

    if i < len(weights) - 1:

        relu_output = f"relu_{i}"

        nodes.append(
            helper.make_node(
                "Relu",
                inputs=[
                    current,
                ],
                outputs=[
                    relu_output,
                ],
                name=f"MLP_ReLU_{i}",
            )
        )

        current = relu_output


# ============================================================
# SOFTMAX
# ============================================================

nodes.append(
    helper.make_node(
        "Softmax",
        inputs=[
            current,
        ],
        outputs=[
            output_name,
        ],
        axis=1,
        name="Output_Softmax",
    )
)


# ============================================================
# CREATE GRAPH
# ============================================================

graph = helper.make_graph(
    nodes=nodes,
    name="AeroTwinEdgeMLP",
    inputs=[
        input_tensor,
    ],
    outputs=[
        output_tensor,
    ],
    initializer=initializers,
)


# ============================================================
# CREATE ONNX MODEL
# ============================================================

model = helper.make_model(
    graph,
    producer_name="AeroTwin Edge",
)


# ============================================================
# IMPORTANT:
# QUALCOMM AI HUB WORKBENCH CHECKER EXPECTS
# ONNX IR VERSION 13 OR LOWER
# ============================================================

model.ir_version = 13


# ============================================================
# ONNX OPSET
# ============================================================

model.opset_import[0].version = 17


# ============================================================
# SAVE MODEL
# ============================================================

ONNX_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

onnx.save(
    model,
    ONNX_PATH,
)


# ============================================================
# LOCAL ONNX VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("Running ONNX checker...")
print("=" * 70)

onnx.checker.check_model(
    model,
    full_check=True,
)

print("ONNX checker: PASS")


# ============================================================
# PRINT IR / OPSET
# ============================================================

print("\nONNX compatibility information:")
print(f"IR version     : {model.ir_version}")

for opset in model.opset_import:
    print(
        f"Opset version  : {opset.version}"
    )


# ============================================================
# INPUT INSPECTION
# ============================================================

print("\nInputs:")

for inp in model.graph.input:

    shape = []

    for dim in inp.type.tensor_type.shape.dim:

        if dim.dim_value:
            shape.append(dim.dim_value)

        elif dim.dim_param:
            shape.append(dim.dim_param)

        else:
            shape.append("?")

    print(
        f"  {inp.name}: "
        f"shape={shape}, "
        f"dtype={inp.type.tensor_type.elem_type}"
    )


# ============================================================
# OUTPUT INSPECTION
# ============================================================

print("\nOutputs:")

for out in model.graph.output:

    shape = []

    for dim in out.type.tensor_type.shape.dim:

        if dim.dim_value:
            shape.append(dim.dim_value)

        elif dim.dim_param:
            shape.append(dim.dim_param)

        else:
            shape.append("?")

    print(
        f"  {out.name}: "
        f"shape={shape}, "
        f"dtype={out.type.tensor_type.elem_type}"
    )


# ============================================================
# OPERATOR INSPECTION
# ============================================================

print("\nOperators:")

domains = set()

for node in model.graph.node:

    domain = node.domain if node.domain else "ai.onnx"

    domains.add(domain)

    print(
        f"  {node.name:<22} "
        f"{node.op_type:<10} "
        f"domain={domain}"
    )


# ============================================================
# CHECK FOR ai.onnx.ml
# ============================================================

if "ai.onnx.ml" in domains:

    raise RuntimeError(
        "ERROR: ai.onnx.ml operator detected."
    )

print("\nAI ONNX ML domain: NOT USED")


# ============================================================
# CHECK OUTPUT TYPES
# ============================================================

for output in model.graph.output:

    elem_type = output.type.tensor_type.elem_type

    allowed_types = {
        TensorProto.FLOAT,
        TensorProto.FLOAT16,
        TensorProto.INT32,
        TensorProto.INT64,
    }

    if elem_type not in allowed_types:

        raise RuntimeError(
            f"Unsupported output dtype: {elem_type}"
        )


print("Output type check: PASS")


# ============================================================
# FILE INFORMATION
# ============================================================

file_size_kb = ONNX_PATH.stat().st_size / 1024


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("CONVERSION SUCCESSFUL")
print("=" * 70)

print(f"\nONNX file:")
print(ONNX_PATH)

print(f"\nFile size:")
print(f"{file_size_kb:.2f} KB")

print("\nArchitecture:")
print("14 -> 32 -> 16 -> 7")

print("\nInput:")
print("float32 [1, 14]")

print("\nOutput:")
print("float32 [1, 7]")

print("\nIR version:")
print("13")

print("\nOpset:")
print("17")

print("\nQualcomm AI Hub compatibility checks:")
print("  Static input shape : PASS")
print("  Numeric output     : PASS")
print("  No string output   : PASS")
print("  No sequence output : PASS")
print("  No ai.onnx.ml      : PASS")
print("  IR version 13     : PASS")

print("\nReady for Qualcomm AI Hub compilation.")

print("=" * 70)