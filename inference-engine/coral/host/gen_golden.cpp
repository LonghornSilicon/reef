// Writes the golden headers the Coral kernel tests compare against: seeded
// inputs, and the outputs the host reference computes from them.
//
//   gen_golden OUT_DIR
//
// writes OUT_DIR/golden/<test>.hpp for each test in coral/tests/. The tests
// require every kernel output to match these values bit for bit, so the
// reference functions called here are the specification the kernels follow.
#include "inference_engine/int8.hpp"
#include "inference_engine/math.hpp"
#include "inference_engine/model.hpp"
#include "inference_engine/operator/activation.hpp"
#include "inference_engine/operator/attention.hpp"
#include "inference_engine/operator/block.hpp"
#include "inference_engine/operator/embedding.hpp"
#include "inference_engine/operator/layernorm.hpp"
#include "inference_engine/operator/linear.hpp"
#include "inference_engine/quant.hpp"
#include "inference_engine/tensor.hpp"
#include "inference_engine/weights.hpp"
#include "pack_model.hpp"
#include "reef_kernels/model_blob.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using inference_engine::Tensor;

/// Same sequence on every host, unlike the std:: distributions.
class Rng {
  public:
    explicit Rng(std::uint32_t seed) : state_(seed) {}

    float uniform(float low, float high) {
        constexpr float kUnit = 1.0F / 16777216.0F; // 2^-24
        const auto fraction = static_cast<float>(next() >> 8U) * kUnit;
        return low + ((high - low) * fraction);
    }

    /// count integers drawn evenly from [low, high].
    template <typename T>
    std::vector<T> integers(std::size_t count, std::int32_t low,
                            std::int32_t high) {
        const auto span = static_cast<std::uint32_t>(high - low) + 1U;
        std::vector<T> values(count);
        for (T& value : values) {
            value = static_cast<T>(
                low + static_cast<std::int32_t>((next() >> 8U) % span));
        }
        return values;
    }

    std::vector<float> uniform(std::size_t count, float low, float high) {
        std::vector<float> values(count);
        for (float& value : values) {
            value = uniform(low, high);
        }
        return values;
    }

  private:
    std::uint32_t next() {
        state_ = (state_ * 1664525U) + 1013904223U;
        return state_;
    }

    std::uint32_t state_;
};

/// count evenly spaced values from low to high inclusive.
std::vector<float> sweep(std::size_t count, float low, float high) {
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto t = static_cast<float>(i) / static_cast<float>(count - 1);
        values[i] = low + ((high - low) * t);
    }
    return values;
}

/// Transpose a [rows, columns] matrix's values into [columns, rows].
template <typename Scalar>
std::vector<Scalar> transposed(const Tensor<Scalar>& matrix) {
    return inference_engine::transpose(matrix, 0, 1).values;
}

std::string float_literal(float value) {
    if (!std::isfinite(value)) {
        throw std::runtime_error("golden values must be finite");
    }
    std::ostringstream text;
    text << std::hexfloat << value << 'F';
    return text.str();
}

/// One generated header: arrays of inputs and expected outputs, then the
/// case tables that point at them.
class Header {
  public:
    explicit Header(std::string test) : test_(std::move(test)) {}

    /// Emit a float array and return its name.
    std::string floats(const std::string& name,
                       const std::vector<float>& values) {
        std::vector<std::string> literals;
        literals.reserve(values.size());
        for (const float value : values) {
            literals.push_back(float_literal(value));
        }
        return array("float", name, literals);
    }

    /// Emit a 16-byte aligned array of 32-bit words and return its name.
    std::string words(const std::string& name,
                      const std::vector<std::uint32_t>& values) {
        std::vector<std::string> literals;
        literals.reserve(values.size());
        for (const std::uint32_t value : values) {
            literals.push_back(std::to_string(value) + "U");
        }
        return array("std::uint32_t", name, literals, "alignas(16) ");
    }

    /// Emit an int8 array and return its name.
    std::string int8s(const std::string& name,
                      const std::vector<std::int8_t>& values) {
        return integers("std::int8_t", name, values);
    }

    /// Emit an array of any integer type and return its name.
    template <typename T>
    std::string integers(const std::string& type, const std::string& name,
                         const std::vector<T>& values) {
        std::vector<std::string> literals;
        literals.reserve(values.size());
        for (const T value : values) {
            literals.push_back(std::to_string(static_cast<long long>(value)));
        }
        return array(type, name, literals);
    }

    /// Emit an array from a fixed-size table and return its name.
    template <typename T, std::size_t N>
    std::string integers(const std::string& type, const std::string& name,
                         const std::array<T, N>& values) {
        return integers(type, name,
                        std::vector<T>(values.begin(), values.end()));
    }

    /// Emit a table of cases, one brace initializer per case.
    void table(const std::string& type, const std::string& name,
               const std::vector<std::string>& cases) {
        body_ << "inline constexpr " << type << ' ' << name << "[] = {\n";
        for (const std::string& entry : cases) {
            body_ << "    {" << entry << "},\n";
        }
        body_ << "};\n\n";
    }

    /// Emit a size constant.
    void constant(const std::string& name, std::size_t value) {
        body_ << "inline constexpr std::size_t " << name << " = " << value
              << ";\n";
    }

    void write(const std::filesystem::path& directory) const {
        std::filesystem::create_directories(directory);
        std::ofstream file(directory / (test_ + ".hpp"));
        file << "// Generated by coral/host/gen_golden.cpp from the host "
                "reference. Do not edit.\n"
             << "#pragma once\n\n#include \"cases.hpp\"\n\n"
             << "#include <cstddef>\n#include <cstdint>\n\n"
             << "namespace golden {\n\n"
             << body_.str() << "\n} // namespace golden\n";
        if (!file) {
            throw std::runtime_error("cannot write " + test_ + ".hpp");
        }
    }

  private:
    std::string array(const std::string& type, const std::string& name,
                      const std::vector<std::string>& literals,
                      const std::string& prefix = "") {
        constexpr std::size_t kPerLine = 6;
        body_ << prefix << "inline constexpr " << type << ' ' << name
              << "[] = {";
        for (std::size_t i = 0; i < literals.size(); ++i) {
            body_ << (i % kPerLine == 0 ? "\n    " : " ") << literals[i] << ',';
        }
        body_ << "\n};\n";
        return name;
    }

    std::string test_;
    std::ostringstream body_;
};

/// Join values with ", " for a brace initializer.
std::string fields(const std::vector<std::string>& values) {
    std::string joined;
    for (const std::string& value : values) {
        joined += (joined.empty() ? "" : ", ") + value;
    }
    return joined;
}

std::string str(std::size_t value) { return std::to_string(value); }

/// Inputs that probe the edges of exp_nonpositive's domain.
std::vector<float> exp_inputs(Rng& rng) {
    std::vector<float> x = sweep(2001, -100.0F, 0.0F);
    const std::vector<float> extra = rng.uniform(1000, -30.0F, 0.0F);
    x.insert(x.end(), extra.begin(), extra.end());
    constexpr float kMin = -87.0F;
    x.insert(x.end(),
             {kMin, std::nextafter(kMin, -1000.0F), std::nextafter(kMin, 0.0F),
              -0.0F, 0.0F, -1e-30F, -std::numeric_limits<float>::denorm_min(),
              -0.34657359F, -0.34657360F, -1.0F, -2.0F, -10.0F});
    return x;
}

/// Inputs for tanh and GELU, both signs.
std::vector<float> signed_inputs(Rng& rng, float range) {
    std::vector<float> x = sweep(2001, -range, range);
    const std::vector<float> extra = rng.uniform(1000, -6.0F, 6.0F);
    x.insert(x.end(), extra.begin(), extra.end());
    x.insert(x.end(), {-0.0F, 0.0F, 1e-8F, -1e-8F, 44.0F, -44.0F, 100.0F,
                       -100.0F, 0.5F, -0.5F});
    return x;
}

void write_math(const std::filesystem::path& directory) {
    Rng rng(1);
    Header header("math");
    std::size_t scratch = 0;
    const auto unary = [&](const std::string& prefix,
                           const std::vector<float>& x,
                           const std::vector<float>& y) {
        scratch = std::max(scratch, x.size());
        return fields({str(x.size()), header.floats(prefix + "_x", x),
                       header.floats(prefix + "_y", y)});
    };

    const std::vector<float> exp_x = exp_inputs(rng);
    std::vector<float> exp_y;
    exp_y.reserve(exp_x.size());
    for (const float value : exp_x) {
        exp_y.push_back(inference_engine::exp_nonpositive(value));
    }
    header.table("UnaryCase", "kExp", {unary("exp", exp_x, exp_y)});

    const std::vector<float> tanh_x = signed_inputs(rng, 12.0F);
    std::vector<float> tanh_y;
    tanh_y.reserve(tanh_x.size());
    for (const float value : tanh_x) {
        tanh_y.push_back(inference_engine::tanh_float(value));
    }
    header.table("UnaryCase", "kTanh", {unary("tanh", tanh_x, tanh_y)});

    const std::vector<float> gelu_x = signed_inputs(rng, 12.0F);
    const Tensor<float> gelu_y =
        inference_engine::gelu(Tensor<float>{{gelu_x.size()}, gelu_x});
    header.table("UnaryCase", "kGelu", {unary("gelu", gelu_x, gelu_y.values)});

    header.constant("kScratchFloats", scratch);
    header.write(directory);
}

/// [rows, in] by [out, in] shapes the linear tests cover: odd sizes that
/// leave partial vector strips, and model-like widths.
struct LinearShape {
    std::size_t rows, in, out;
};
constexpr LinearShape kLinearShapes[] = {
    {1, 7, 5}, {3, 37, 21}, {2, 64, 48}, {1, 256, 64}};

void write_linear(const std::filesystem::path& directory) {
    Rng rng(2);
    Header header("linear");
    std::size_t scratch = 0;
    std::size_t scratch_int8 = 0;
    std::vector<std::string> plain;
    std::vector<std::string> biased;
    std::vector<std::string> w8a8;
    std::vector<std::string> w8a8_biased;

    std::size_t index = 0;
    for (const LinearShape& shape : kLinearShapes) {
        const std::string p = "linear" + str(index++);
        const Tensor<float> x{{shape.rows, shape.in},
                              rng.uniform(shape.rows * shape.in, -2.0F, 2.0F)};
        Tensor<float> w{{shape.out, shape.in},
                        rng.uniform(shape.out * shape.in, -0.5F, 0.5F)};
        const Tensor<float> bias{{shape.out},
                                 rng.uniform(shape.out, -1.0F, 1.0F)};
        // A zero input row exercises the W8A8 all-zero scale.
        Tensor<float> x_zero_row = x;
        std::fill_n(x_zero_row.values.begin(), shape.in, 0.0F);

        scratch = std::max(scratch, shape.rows * shape.out);
        scratch_int8 = std::max(scratch_int8, shape.in);
        const std::string dims =
            fields({str(shape.rows), str(shape.in), str(shape.out)});
        const std::string xs = header.floats(p + "_x", x.values);
        const std::string xz = header.floats(p + "_xz", x_zero_row.values);
        const std::string wt = header.floats(p + "_wt", transposed(w));
        const std::string b = header.floats(p + "_bias", bias.values);

        plain.push_back(fields(
            {dims, xs, wt,
             header.floats(p + "_y",
                           inference_engine::linear<float>(x, w).values)}));
        biased.push_back(
            fields({dims, xs, wt, b,
                    header.floats(p + "_yb", inference_engine::linear<float>(
                                                 x, w, std::cref(bias))
                                                 .values)}));

        const inference_engine::QuantizedWeight<float> q =
            inference_engine::quantize_rows(w);
        const std::string qt = header.int8s(p + "_qt", transposed(q.values));
        const std::string qs = header.floats(p + "_qs", q.scales);
        // Each input once as drawn and once with its first row zeroed.
        const auto add_w8a8 = [&](const std::string& input_name,
                                  const Tensor<float>& input,
                                  const std::string& output_prefix) {
            w8a8.push_back(fields(
                {dims, input_name, qt, qs,
                 header.floats(
                     output_prefix,
                     inference_engine::linear<float>(input, q).values)}));
            w8a8_biased.push_back(
                fields({dims, input_name, qt, qs, b,
                        header.floats(output_prefix + "b",
                                      inference_engine::linear<float>(
                                          input, q, std::cref(bias))
                                          .values)}));
        };
        add_w8a8(xs, x, p + "_yq");
        add_w8a8(xz, x_zero_row, p + "_yqz");
    }
    header.table("LinearCase", "kLinear", plain);
    header.table("LinearBiasCase", "kLinearBias", biased);
    header.table("W8a8Case", "kW8a8", w8a8);
    header.table("W8a8BiasCase", "kW8a8Bias", w8a8_biased);

    // matvec_rows_f32: one row against an [out][in] matrix, as lm_head.
    std::vector<std::string> matvec;
    for (const auto& [out, in] :
         {std::pair<std::size_t, std::size_t>{33, 19}, {64, 64}, {100, 48}}) {
        const std::string p = "matvec" + str(index++);
        const Tensor<float> x{{1, in}, rng.uniform(in, -2.0F, 2.0F)};
        const Tensor<float> w{{out, in}, rng.uniform(out * in, -0.5F, 0.5F)};
        scratch = std::max(scratch, out);
        matvec.push_back(fields(
            {str(out), str(in), header.floats(p + "_w", w.values),
             header.floats(p + "_x", x.values),
             header.floats(p + "_y",
                           inference_engine::linear<float>(x, w).values)}));
    }
    header.table("MatvecCase", "kMatvec", matvec);

    header.constant("kScratchFloats", scratch);
    header.constant("kScratchInt8", scratch_int8);
    header.write(directory);
}

void write_layernorm(const std::filesystem::path& directory) {
    Rng rng(3);
    Header header("layernorm");
    constexpr float kEps = 1e-5F;
    std::size_t scratch = 0;
    std::vector<std::string> cases;
    std::size_t index = 0;
    for (const auto& [rows, width] :
         {std::pair<std::size_t, std::size_t>{3, 37},
          {1, 64},
          {2, 256},
          {1, 1}}) {
        const std::string p = "ln" + str(index++);
        Tensor<float> x{{rows, width}, rng.uniform(rows * width, -3.0F, 3.0F)};
        for (float& value : x.values) {
            value += 1.5F; // a nonzero mean
        }
        const Tensor<float> gain{{width}, rng.uniform(width, 0.5F, 1.5F)};
        const Tensor<float> shift{{width}, rng.uniform(width, -0.5F, 0.5F)};
        const Tensor<float> y =
            inference_engine::layernorm<float>(x, gain, std::cref(shift), kEps);
        scratch = std::max(scratch, rows * width);
        cases.push_back(fields({str(rows), str(width), float_literal(kEps),
                                header.floats(p + "_x", x.values),
                                header.floats(p + "_gain", gain.values),
                                header.floats(p + "_shift", shift.values),
                                header.floats(p + "_y", y.values)}));
    }
    header.table("LayerNormCase", "kLayerNorm", cases);
    header.constant("kScratchFloats", scratch);
    header.write(directory);
}

struct AttentionShape {
    std::size_t query_len, key_len, head_dim, query_start, window;
};
// window 0 is global attention.
constexpr AttentionShape kAttentionShapes[] = {
    {1, 1, 8, 0, 0},    {1, 7, 8, 6, 0},    {5, 5, 16, 0, 0},   {4, 9, 4, 5, 3},
    {1, 20, 16, 19, 4}, {3, 40, 32, 37, 0}, {2, 40, 32, 38, 16}};

void write_attention(const std::filesystem::path& directory) {
    Rng rng(4);
    Header header("attention");
    std::size_t scratch = 0;
    std::size_t scores = 0;
    std::vector<std::string> cases;
    std::size_t index = 0;
    for (const AttentionShape& s : kAttentionShapes) {
        const std::string p = "attn" + str(index++);
        const Tensor<float> q{
            {s.query_len, s.head_dim},
            rng.uniform(s.query_len * s.head_dim, -2.0F, 2.0F)};
        const Tensor<float> k{{s.key_len, s.head_dim},
                              rng.uniform(s.key_len * s.head_dim, -2.0F, 2.0F)};
        const Tensor<float> v{{s.key_len, s.head_dim},
                              rng.uniform(s.key_len * s.head_dim, -2.0F, 2.0F)};
        const std::optional<std::size_t> window =
            s.window > 0 ? std::optional<std::size_t>(s.window) : std::nullopt;
        const Tensor<float> y = inference_engine::attend_head(
            q, k, v,
            inference_engine::causal_visibility(s.query_len, s.key_len,
                                                s.query_start, window));
        scratch = std::max(scratch, s.query_len * s.head_dim);
        scores = std::max(scores, s.key_len);
        cases.push_back(
            fields({str(s.query_len), str(s.key_len), str(s.head_dim),
                    str(s.query_start), str(s.window),
                    header.floats(p + "_q", q.values),
                    header.floats(p + "_k", k.values),
                    header.floats(p + "_v", v.values),
                    header.floats(p + "_y", y.values)}));
    }
    header.table("AttentionCase", "kAttention", cases);
    header.constant("kScratchFloats", scratch);
    header.constant("kScratchScores", scores);
    header.write(directory);
}

void write_elementwise(const std::filesystem::path& directory) {
    Rng rng(5);
    Header header("elementwise");
    std::size_t scratch = 0;

    std::vector<std::string> adds;
    for (const std::size_t n : {std::size_t{37}, std::size_t{100}}) {
        const std::string p = "add" + str(n);
        const Tensor<float> a{{1, n}, rng.uniform(n, -4.0F, 4.0F)};
        const Tensor<float> b{{1, n}, rng.uniform(n, -4.0F, 4.0F)};
        scratch = std::max(scratch, n);
        adds.push_back(fields(
            {str(n), header.floats(p + "_a", a.values),
             header.floats(p + "_b", b.values),
             header.floats(p + "_y",
                           inference_engine::residual_add(a, b).values)}));
    }
    header.table("AddCase", "kAdd", adds);

    constexpr std::size_t kVocab = 10;
    constexpr std::size_t kWidth = 37;
    constexpr std::size_t kPositions = 8;
    const Tensor<float> wte{{kVocab, kWidth},
                            rng.uniform(kVocab * kWidth, -1.0F, 1.0F)};
    const Tensor<float> wpe{{kPositions, kWidth},
                            rng.uniform(kPositions * kWidth, -1.0F, 1.0F)};
    const std::string wte_name = header.floats("emb_wte", wte.values);
    const std::string wpe_name = header.floats("emb_wpe", wpe.values);
    std::vector<std::string> embeddings;
    for (const auto& [id, position] :
         {std::pair<std::uint32_t, std::size_t>{3, 5}, {9, 0}, {0, 7}}) {
        const Tensor<float> y = inference_engine::add_position_embeddings(
            inference_engine::lookup_token_embeddings({id}, wte), wpe,
            position);
        embeddings.push_back(
            fields({str(kWidth), str(id), str(position), wte_name, wpe_name,
                    header.floats("emb" + str(id) + "_y", y.values)}));
    }
    scratch = std::max(scratch, kWidth);
    header.table("EmbeddingCase", "kEmbedding", embeddings);

    // Ties must resolve to the first maximum, including +0 against -0.
    std::vector<float> random = rng.uniform(1000, -10.0F, 10.0F);
    std::vector<float> ties = rng.uniform(50, -1.0F, 1.0F);
    ties[7] = 5.0F;
    ties[30] = 5.0F;
    std::vector<float> zeros = {-1.0F, -0.0F, 0.0F, -0.0F};
    std::vector<std::string> argmaxes;
    std::size_t index = 0;
    for (const std::vector<float>* x : {&random, &ties, &zeros}) {
        const std::size_t best =
            inference_engine::argmax(Tensor<float>{{x->size()}, *x});
        argmaxes.push_back(fields(
            {str(x->size()), header.floats("argmax" + str(index++) + "_x", *x),
             str(best)}));
    }
    header.table("ArgmaxCase", "kArgmax", argmaxes);

    header.constant("kScratchFloats", scratch);
    header.write(directory);
}

/// The integer-only operators of int8.hpp. Shapes include the ones the
/// original standalone int8 kernels ran (16 x 64), odd sizes that end
/// vector strips early, and shifts of 1, where half the sums are ties.
void write_int8(const std::filesystem::path& directory) {
    using inference_engine::Tensor;
    Rng rng(6);
    Header header("int8");
    std::size_t scratch = 0;
    std::size_t index = 0;
    const auto name = [&](const std::string& what) {
        return "i8_" + str(index) + "_" + what;
    };
    const auto int8_data = [&](std::size_t count) {
        return rng.integers<std::int8_t>(count, -128, 127);
    };

    // linear_int8, with and without a bias.
    struct Shape {
        std::size_t rows, in, out;
        unsigned shift;
    };
    std::vector<std::string> linear;
    std::vector<std::string> linear_bias;
    for (const Shape& s : {Shape{1, 7, 5, 1}, Shape{3, 37, 21, 6},
                           Shape{16, 64, 64, 8}, Shape{1, 256, 64, 10}}) {
        const Tensor<std::int8_t> x{{s.rows, s.in}, int8_data(s.rows * s.in)};
        const Tensor<std::int8_t> w{{s.out, s.in}, int8_data(s.out * s.in)};
        const Tensor<std::int32_t> bias{
            {s.out}, rng.integers<std::int32_t>(s.out, -4096, 4096)};
        const std::string dims =
            fields({str(s.rows), str(s.in), str(s.out), str(s.shift)});
        const std::string xs = header.int8s(name("x"), x.values);
        const std::string wt = header.int8s(name("wt"), transposed(w));
        linear.push_back(
            fields({dims, xs, wt,
                    header.int8s(
                        name("y"),
                        inference_engine::linear_int8(x, w, s.shift).values)}));
        linear_bias.push_back(
            fields({dims, xs, wt,
                    header.integers("std::int32_t", name("bias"), bias.values),
                    header.int8s(name("yb"), inference_engine::linear_int8(
                                                 x, w, s.shift, std::cref(bias))
                                                 .values)}));
        scratch = std::max(scratch, s.rows * s.out);
        ++index;
    }
    header.table("Int8LinearCase", "kInt8Linear", linear);
    header.table("Int8LinearBiasCase", "kInt8LinearBias", linear_bias);

    // gelu_int8: every int8 input, then random ones, through two tables.
    std::vector<std::string> gelus;
    for (const auto& [in_scale, out_scale] :
         {std::pair{0.05F, 0.05F}, std::pair{0.1F, 0.02F}}) {
        std::vector<std::int8_t> x(256);
        for (std::size_t i = 0; i < x.size(); ++i) {
            x[i] = static_cast<std::int8_t>(static_cast<int>(i) - 128);
        }
        const std::vector<std::int8_t> extra = int8_data(1000);
        x.insert(x.end(), extra.begin(), extra.end());
        const inference_engine::Int8Table table =
            inference_engine::gelu_table(in_scale, out_scale);
        const Tensor<std::int8_t> y = inference_engine::gelu_int8(
            Tensor<std::int8_t>{{x.size()}, x}, table);
        gelus.push_back(fields(
            {str(x.size()),
             header.integers("std::int8_t", name("table"), table),
             header.int8s(name("x"), x), header.int8s(name("y"), y.values)}));
        scratch = std::max(scratch, x.size());
        ++index;
    }
    header.table("Int8TableCase", "kInt8Gelu", gelus);

    const inference_engine::ExpTable exp_table = inference_engine::exp_table();
    const std::string exp_name =
        header.integers("std::uint16_t", "i8_exp_table", exp_table);

    // softmax_int8: the original 16 x 64 at shift 4, a wide spread at
    // shift 0, and rows of equal scores.
    struct SoftmaxShape {
        std::size_t rows, n;
        unsigned shift;
        std::int32_t spread;
    };
    std::vector<std::string> softmaxes;
    std::size_t score_scratch = 0;
    for (const SoftmaxShape& s :
         {SoftmaxShape{16, 64, 4, 2048}, SoftmaxShape{3, 37, 0, 3000},
          SoftmaxShape{2, 5, 2, 0}}) {
        const Tensor<std::int32_t> scores{
            {s.rows, s.n},
            rng.integers<std::int32_t>(s.rows * s.n, -s.spread, s.spread)};
        const Tensor<bool> all{{s.rows, s.n},
                               std::vector<bool>(s.rows * s.n, true)};
        const Tensor<std::uint8_t> probs =
            inference_engine::masked_softmax_int8(scores, all, s.shift,
                                                  exp_table);
        softmaxes.push_back(fields(
            {str(s.rows), str(s.n), str(s.shift), exp_name,
             header.integers("std::int32_t", name("scores"), scores.values),
             header.integers("std::uint8_t", name("probs"), probs.values)}));
        score_scratch = std::max(score_scratch, s.rows * s.n);
        ++index;
    }
    header.table("Int8SoftmaxCase", "kInt8Softmax", softmaxes);

    // attention_head_int8: the float cases' shapes plus the original
    // 16-position, 64-wide head.
    struct HeadShape {
        std::size_t query_len, key_len, head_dim, query_start, window;
        unsigned score_shift;
    };
    std::vector<std::string> heads;
    std::size_t key_scratch = 0;
    for (const HeadShape& s :
         {HeadShape{1, 1, 8, 0, 0, 6}, HeadShape{1, 7, 8, 6, 0, 6},
          HeadShape{5, 5, 16, 0, 0, 7}, HeadShape{4, 9, 4, 5, 3, 4},
          HeadShape{1, 20, 16, 19, 4, 7}, HeadShape{3, 40, 32, 37, 0, 9},
          HeadShape{16, 16, 64, 0, 0, 10}}) {
        constexpr unsigned kOutShift = 8;
        const Tensor<std::int8_t> q{{s.query_len, s.head_dim},
                                    int8_data(s.query_len * s.head_dim)};
        const Tensor<std::int8_t> k{{s.key_len, s.head_dim},
                                    int8_data(s.key_len * s.head_dim)};
        const Tensor<std::int8_t> v{{s.key_len, s.head_dim},
                                    int8_data(s.key_len * s.head_dim)};
        const std::optional<std::size_t> window =
            s.window > 0 ? std::optional<std::size_t>(s.window) : std::nullopt;
        const Tensor<std::int8_t> y = inference_engine::attend_head_int8(
            q, k, v,
            inference_engine::causal_visibility(s.query_len, s.key_len,
                                                s.query_start, window),
            s.score_shift, exp_table, kOutShift);
        heads.push_back(
            fields({str(s.query_len), str(s.key_len), str(s.head_dim),
                    str(s.query_start), str(s.window), str(s.score_shift),
                    str(kOutShift), exp_name, header.int8s(name("q"), q.values),
                    header.int8s(name("k"), k.values),
                    header.int8s(name("v"), v.values),
                    header.int8s(name("y"), y.values)}));
        scratch = std::max(scratch, s.query_len * s.head_dim);
        key_scratch = std::max(key_scratch, s.key_len);
        ++index;
    }
    header.table("Int8AttentionCase", "kInt8Attention", heads);

    // layernorm_int8: gains near 1.0 (32 in Q5) and small shifts, as the
    // original kernel used, at power-of-two and other widths.
    std::vector<std::string> norms;
    for (const auto& [rows, width] :
         {std::pair<std::size_t, std::size_t>{3, 37},
          {16, 64},
          {2, 256},
          {1, 1}}) {
        const Tensor<std::int8_t> x{{rows, width}, int8_data(rows * width)};
        const Tensor<std::int8_t> gain{
            {width}, rng.integers<std::int8_t>(width, 16, 48)};
        const Tensor<std::int8_t> shift{
            {width}, rng.integers<std::int8_t>(width, -16, 15)};
        const Tensor<std::int8_t> y =
            inference_engine::layernorm_int8(x, gain, shift);
        norms.push_back(
            fields({str(rows), str(width), header.int8s(name("x"), x.values),
                    header.int8s(name("gain"), gain.values),
                    header.int8s(name("shift"), shift.values),
                    header.int8s(name("y"), y.values)}));
        scratch = std::max(scratch, rows * width);
        ++index;
    }
    header.table("Int8LayerNormCase", "kInt8LayerNorm", norms);

    // add_int8 saturates; full-range addends overflow often.
    std::vector<std::string> adds;
    for (const std::size_t n : {std::size_t{37}, std::size_t{100}}) {
        const Tensor<std::int8_t> a{{n}, int8_data(n)};
        const Tensor<std::int8_t> b{{n}, int8_data(n)};
        adds.push_back(
            fields({str(n), header.int8s(name("a"), a.values),
                    header.int8s(name("b"), b.values),
                    header.int8s(name("y"),
                                 inference_engine::add_int8(a, b).values)}));
        scratch = std::max(scratch, n);
        ++index;
    }
    header.table("Int8AddCase", "kInt8Add", adds);

    // embedding_int8 = lookup_token_embeddings for both tables, then
    // add_int8.
    constexpr std::size_t kVocab = 10;
    constexpr std::size_t kWidth = 37;
    constexpr std::size_t kPositions = 8;
    const Tensor<std::int8_t> wte{{kVocab, kWidth}, int8_data(kVocab * kWidth)};
    const Tensor<std::int8_t> wpe{{kPositions, kWidth},
                                  int8_data(kPositions * kWidth)};
    const std::string wte_name = header.int8s("i8_wte", wte.values);
    const std::string wpe_name = header.int8s("i8_wpe", wpe.values);
    std::vector<std::string> embeddings;
    for (const auto& [id, position] :
         {std::pair<std::uint32_t, std::uint32_t>{3, 5}, {9, 0}, {0, 7}}) {
        const Tensor<std::int8_t> y = inference_engine::add_int8(
            inference_engine::lookup_token_embeddings({id}, wte),
            inference_engine::lookup_token_embeddings({position}, wpe));
        embeddings.push_back(
            fields({str(kWidth), str(id), str(position), wte_name, wpe_name,
                    header.int8s(name("y"), y.values)}));
        ++index;
    }
    scratch = std::max(scratch, kWidth);
    header.table("Int8EmbeddingCase", "kInt8Embedding", embeddings);

    header.constant("kScratchInt8", scratch);
    header.constant("kScratchScores", std::max(score_scratch, key_scratch));
    header.write(directory);
}

/// A small GPT-Neo with every structural feature of the real ones: global
/// and local layers, a window shorter than the sequence, several heads.
inference_engine::GptNeoConfig small_config() {
    inference_engine::GptNeoConfig config;
    config.vocab_size = 96;
    config.max_positions = 32;
    config.hidden_size = 32;
    config.num_layers = 4;
    config.num_heads = 4;
    config.inner_size = 128;
    config.window_size = 8;
    return config;
}

void write_gpt_neo(const std::filesystem::path& directory) {
    constexpr std::uint32_t kSeed = 7;
    constexpr std::size_t kNewTokens = 20;
    const inference_engine::TokenIds prompt = {5, 17, 42, 3, 77};
    Header header("gpt_neo");
    std::vector<std::string> cases;

    inference_engine::ModelWeights<float> weights =
        inference_engine::random_gpt_neo(small_config(), kSeed);
    for (const auto precision :
         {reef_kernels::Precision::kFloat32, reef_kernels::Precision::kW8A8}) {
        const bool w8a8 = precision == reef_kernels::Precision::kW8A8;
        if (w8a8) {
            inference_engine::quantize_w8a8(weights);
        }
        const std::string p = w8a8 ? "w8a8" : "fp32";

        // GPTNeo::generate, keeping the logits behind each token.
        inference_engine::GPTNeo<float> model(weights);
        std::vector<float> logits;
        inference_engine::TokenIds tokens;
        Tensor<float> step = model.forward(prompt);
        while (true) {
            logits.insert(logits.end(), step.values.begin(), step.values.end());
            tokens.push_back(
                static_cast<std::uint32_t>(inference_engine::argmax(step)));
            if (tokens.size() == kNewTokens) {
                break;
            }
            step = model.forward({tokens.back()});
        }
        inference_engine::GPTNeo<float> check(weights);
        if (check.generate(prompt, kNewTokens) != tokens) {
            throw std::runtime_error("generate disagrees with forward");
        }

        const std::vector<std::uint8_t> blob =
            coral_host::pack_gpt_neo(weights, precision);
        std::vector<std::uint32_t> blob_words(blob.size() / 4);
        std::memcpy(blob_words.data(), blob.data(), blob.size());
        cases.push_back(
            fields({header.words(p + "_blob", blob_words), str(prompt.size()),
                    header.words(p + "_prompt", prompt), str(kNewTokens),
                    header.words(p + "_tokens", tokens),
                    str(weights.config.vocab_size),
                    header.floats(p + "_logits", logits)}));
    }
    header.table("ModelCase", "kGptNeo", cases);
    header.constant("kMaxTokens", prompt.size() + kNewTokens);
    header.write(directory);
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            std::cerr << "usage: gen_golden OUT_DIR\n";
            return 2;
        }
        const std::filesystem::path directory =
            std::filesystem::path(argv[1]) / "golden";
        write_math(directory);
        write_linear(directory);
        write_layernorm(directory);
        write_attention(directory);
        write_elementwise(directory);
        write_int8(directory);
        write_gpt_neo(directory);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gen_golden: " << error.what() << '\n';
        return 1;
    }
}
