# TinyStories-28M Quantization Evaluation

## Overview

This experiment evaluates how different weight and activation precisions affect the text generation quality of **TinyStories-Instruct-28M**.

We compare 10 precision configurations across 5 story-generation prompts to identify promising quantization formats for efficient LLM inference on custom hardware.

## Precision Configurations

| Configuration | Weight Precision | Activation Precision |
|---|---|---|
| FP32 | 32-bit | 32-bit |
| INT8 / W8A8 | 8-bit | 8-bit |
| W8A16 | 8-bit | 16-bit |
| W8A32 | 8-bit | 32-bit |
| INT4 | 4-bit | Implementation-dependent |
| W4A8 | 4-bit | 8-bit |
| W4A16 | 4-bit | 16-bit |
| W8A4 | 8-bit | 4-bit |
| W4A4 | 4-bit | 4-bit |

**Note:** INT8 and W8A8 are listed separately in the experiment outputs but produce identical stories across all five prompts. The same is true for INT4 and W4A16 in four of five prompts. Exact implementation details should be verified before assuming equivalent arithmetic.

## Results

Five story prompts were evaluated:

1. Lily and the lost kitten
2. Tom and the sandcastle
3. Ben and the moon
4. Anna and the balloon/rock
5. Mia and the lantern

### Generation Quality

| Precision | Kitten | Sandcastle | Moon | Balloon | Lantern |
|---|---|---|---|---|---|
| FP32 | Good | Good | Good | Fair | Good |
| INT8 / W8A8 | Good | Good | Good | Fair | Good |
| W8A16 | Good | Good | Good | Fair | Good |
| W8A32 | Good | Good | Good | Fair | Good |
| INT4 | Poor | Good | Good | Poor | Good |
| W4A8 | Fair | Fair | Good | Fair | Good |
| W4A16 | Poor | Fair | Good | Poor | Good |
| W8A4 | Fail | Fail | Fail | Fail | Fail |
| W4A4 | Fail | Fail | Fail | Fail | Fail |

**Rating definitions:**
- **Good:** Coherent and understandable narrative.
- **Fair:** Understandable but contains repetition or inconsistencies.
- **Poor:** Significant narrative errors or contradictions.
- **Fail:** Largely incoherent or repetitive output.

Ratings are qualitative and based on one generation per prompt and configuration.

## Key Findings

### 1. W8A8 preserves generation quality

- Produces coherent stories across all 5 prompts.
- Output quality is comparable to FP32 in these examples.
- A promising baseline for low-precision inference.

### 2. W4A8 is a viable candidate for further evaluation

- All 5 outputs remain at least partially understandable.
- Some narrative degradation and repetition occur.
- Reduces raw weight storage by approximately **50% compared with W8A8**, excluding quantization metadata.

### 3. Four-bit activations cause severe degradation

- W8A4 and W4A4 produce incoherent outputs on **5/5 prompts**.
- W4A8 performs substantially better than W8A4.
- Suggests activation precision is particularly sensitive in the current quantization setup.

### 4. Four-bit weight quantization is less consistent

- INT4 and W4A16 produce coherent stories for some prompts but significant errors for others.
- Increasing activation precision does not consistently restore FP32-like generation quality.

## Hardware Implications

| Configuration | Observed Quality | Recommendation |
|---|---|---|
| FP32 | Good | Reference baseline |
| W8A8 | Good | Primary quantized baseline |
| W8A16 | Good | Higher-precision comparison |
| W4A8 | Fair–Good | Investigate weight compression |
| W4A16 | Variable | Secondary candidate |
| W8A4 | Failed | Investigate activation quantization |
| W4A4 | Failed | Not currently promising |

**Recommended initial focus: FP32, W8A8, W8A16, and W4A8.**

These configurations offer useful comparisons between generation quality, weight storage, activation precision, and potential hardware efficiency.

## Next Steps

1. **Quantitative quality evaluation:** Measure perplexity and generation quality across a larger dataset.
2. **Controlled comparisons:** Use consistent prompts, decoding settings, and quantization calibration.
3. **Performance profiling:** Measure execution cycles, memory traffic, and latency for each configuration.
4. **Hardware tradeoff analysis:** Compare quality against memory footprint, bandwidth requirements, and compute efficiency.

## Conclusion

Across five story-generation prompts, **8-bit weights and activations preserve good generation quality**, while 4-bit weights introduce moderate and inconsistent degradation.

The strongest observation is that **4-bit activation quantization causes severe quality loss**, even when weights remain at 8-bit precision.

For the current TinyStories-28M inference setup, **W8A8 is the most promising low-precision baseline**, with **W4A8 worth investigating for additional weight compression**.

These results are preliminary qualitative observations, not definitive model accuracy measurements or hardware performance results.