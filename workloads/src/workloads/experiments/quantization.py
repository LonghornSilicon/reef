"""The precisions experiments compare and which operators each one quantizes."""

# int8 rounds Linear weights (per output channel) and inputs (per token);
# int4 rounds Linear weights only, in groups of INT4_GROUP inputs.
PRECISIONS = ("fp32", "bf16", "int8", "int4")
INT4_GROUP = 32


def is_quantized(path: str, operator: str) -> bool:
    # lm_head shares its weight with the token embedding, a lookup table,
    # so both stay in floating point.
    return operator == "Linear" and path != "lm_head"
