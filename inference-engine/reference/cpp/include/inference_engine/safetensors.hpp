#pragma once

/** @file
 *  @brief Read and write float tensors in the safetensors format.
 *
 * A safetensors file is an 8-byte little-endian header length, a JSON header
 * naming each tensor's dtype, shape and byte range, then the raw data. Only
 * F32 tensors are read; other dtypes (such as GPT-Neo's boolean attention
 * mask buffers) are skipped. The host is assumed to be little-endian, as
 * every machine this engine targets is.
 */

#include "inference_engine/tensor.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace inference_engine {

/** A parsed JSON value, enough for safetensors headers and config.json. */
struct JsonValue {
    /// Which JSON type the value holds.
    enum class Kind : std::uint8_t {
        kNull,
        kBool,
        kNumber,
        kString,
        kArray,
        kObject
    };

    /// Type of this value.
    Kind kind = Kind::kNull;
    /// Value of a boolean.
    bool boolean = false;
    /// Value of a number.
    double number = 0.0;
    /// Value of a string.
    std::string text;
    /// Elements of an array, or member values of an object.
    std::vector<JsonValue> items;
    /// Member names of an object, parallel to items.
    std::vector<std::string> keys;

    /** Whether an object has a member.
     *
     * @param key Member name.
     * @return Whether this is an object holding @p key.
     */
    [[nodiscard]] bool contains(std::string_view key) const {
        return std::any_of(
            keys.begin(), keys.end(),
            [key](const std::string& name) { return name == key; });
    }

    /** Look up an object member.
     *
     * @param key Member name.
     * @return The member's value.
     * @throws std::invalid_argument If this is not an object holding @p key.
     */
    [[nodiscard]] const JsonValue& at(std::string_view key) const {
        for (std::size_t index = 0; index < keys.size(); ++index) {
            if (keys[index] == key) {
                return items[index];
            }
        }
        throw std::invalid_argument("JSON object has no member \"" +
                                    std::string(key) + "\"");
    }
};

namespace detail {

/** Recursive-descent parser for one JSON document. */
class JsonParser {
  public:
    /** Bind the text to parse; it must outlive the parser.
     *
     * @param text Complete JSON document.
     */
    explicit JsonParser(std::string_view text) : text_(text) {}

    /** Parse the whole document.
     *
     * @return The top-level value.
     * @throws std::invalid_argument If the text is not valid JSON.
     */
    JsonValue parse() {
        JsonValue result = value();
        skip_space();
        if (pos_ != text_.size()) {
            fail("trailing characters");
        }
        return result;
    }

  private:
    [[noreturn]] void fail(const std::string& what) const {
        throw std::invalid_argument("invalid JSON at offset " +
                                    std::to_string(pos_) + ": " + what);
    }

    void skip_space() {
        while (pos_ < text_.size() &&
               (text_[pos_] == ' ' || text_[pos_] == '\t' ||
                text_[pos_] == '\n' || text_[pos_] == '\r')) {
            ++pos_;
        }
    }

    [[nodiscard]] char peek() const {
        if (pos_ >= text_.size()) {
            fail("unexpected end");
        }
        return text_[pos_];
    }

    void expect(char wanted) {
        if (peek() != wanted) {
            fail(std::string("expected '") + wanted + "'");
        }
        ++pos_;
    }

    JsonValue value() {
        skip_space();
        const char next = peek();
        if (next == '{') {
            return object();
        }
        if (next == '[') {
            return array();
        }
        if (next == '"') {
            JsonValue result;
            result.kind = JsonValue::Kind::kString;
            result.text = string();
            return result;
        }
        if (next == '-' || (next >= '0' && next <= '9')) {
            return number();
        }
        return literal();
    }

    JsonValue object() {
        JsonValue result;
        result.kind = JsonValue::Kind::kObject;
        expect('{');
        skip_space();
        if (peek() == '}') {
            ++pos_;
            return result;
        }
        while (true) {
            skip_space();
            result.keys.push_back(string());
            skip_space();
            expect(':');
            result.items.push_back(value());
            skip_space();
            if (peek() == '}') {
                ++pos_;
                return result;
            }
            expect(',');
        }
    }

    JsonValue array() {
        JsonValue result;
        result.kind = JsonValue::Kind::kArray;
        expect('[');
        skip_space();
        if (peek() == ']') {
            ++pos_;
            return result;
        }
        while (true) {
            result.items.push_back(value());
            skip_space();
            if (peek() == ']') {
                ++pos_;
                return result;
            }
            expect(',');
        }
    }

    std::uint32_t hex4() {
        constexpr std::uint32_t kHexBase = 16;
        constexpr std::uint32_t kDecimalDigits = 10;
        std::uint32_t code = 0;
        for (int digit = 0; digit < 4; ++digit) {
            const char hex = peek();
            ++pos_;
            code *= kHexBase;
            if (hex >= '0' && hex <= '9') {
                code += static_cast<std::uint32_t>(hex - '0');
            } else if (hex >= 'a' && hex <= 'f') {
                code += static_cast<std::uint32_t>(hex - 'a') + kDecimalDigits;
            } else if (hex >= 'A' && hex <= 'F') {
                code += static_cast<std::uint32_t>(hex - 'A') + kDecimalDigits;
            } else {
                fail("bad \\u escape");
            }
        }
        return code;
    }

    static void append_utf8(std::string& out, std::uint32_t code) {
        const auto byte = [](std::uint32_t bits) {
            return static_cast<char>(static_cast<unsigned char>(bits));
        };
        if (code < 0x80U) {
            out += byte(code);
        } else if (code < 0x800U) {
            out += byte(0xC0U | (code >> 6U));
            out += byte(0x80U | (code & 0x3FU));
        } else if (code < 0x10000U) {
            out += byte(0xE0U | (code >> 12U));
            out += byte(0x80U | ((code >> 6U) & 0x3FU));
            out += byte(0x80U | (code & 0x3FU));
        } else {
            out += byte(0xF0U | (code >> 18U));
            out += byte(0x80U | ((code >> 12U) & 0x3FU));
            out += byte(0x80U | ((code >> 6U) & 0x3FU));
            out += byte(0x80U | (code & 0x3FU));
        }
    }

    std::uint32_t escaped_code_point() {
        constexpr std::uint32_t kHighFirst = 0xD800U;
        constexpr std::uint32_t kLowFirst = 0xDC00U;
        constexpr std::uint32_t kLowEnd = 0xE000U;
        constexpr std::uint32_t kSupplementary = 0x10000U;
        constexpr std::uint32_t kTenBits = 10;
        const std::uint32_t code = hex4();
        if (code < kHighFirst || code >= kLowFirst) {
            return code;
        }
        // A high surrogate must be followed by \u and a low surrogate.
        expect('\\');
        expect('u');
        const std::uint32_t low = hex4();
        if (low < kLowFirst || low >= kLowEnd) {
            fail("unpaired surrogate");
        }
        return kSupplementary + ((code - kHighFirst) << kTenBits) +
               (low - kLowFirst);
    }

    std::string string() {
        expect('"');
        std::string out;
        while (true) {
            const char next = peek();
            ++pos_;
            if (next == '"') {
                return out;
            }
            if (next != '\\') {
                out += next;
                continue;
            }
            const char escape = peek();
            ++pos_;
            switch (escape) {
            case '"':
            case '\\':
            case '/':
                out += escape;
                break;
            case 'b':
                out += '\b';
                break;
            case 'f':
                out += '\f';
                break;
            case 'n':
                out += '\n';
                break;
            case 'r':
                out += '\r';
                break;
            case 't':
                out += '\t';
                break;
            case 'u':
                append_utf8(out, escaped_code_point());
                break;
            default:
                fail("bad escape");
            }
        }
    }

    JsonValue number() {
        const std::size_t start = pos_;
        while (pos_ < text_.size() &&
               (text_[pos_] == '-' || text_[pos_] == '+' ||
                text_[pos_] == '.' || text_[pos_] == 'e' ||
                text_[pos_] == 'E' ||
                (text_[pos_] >= '0' && text_[pos_] <= '9'))) {
            ++pos_;
        }
        const std::string digits(text_.substr(start, pos_ - start));
        JsonValue result;
        result.kind = JsonValue::Kind::kNumber;
        std::size_t used = 0;
        try {
            result.number = std::stod(digits, &used);
        } catch (const std::logic_error&) {
            fail("bad number");
        }
        if (used != digits.size()) {
            fail("bad number");
        }
        return result;
    }

    JsonValue literal() {
        JsonValue result;
        if (text_.substr(pos_, 4) == "true") {
            result.kind = JsonValue::Kind::kBool;
            result.boolean = true;
            pos_ += 4;
        } else if (text_.substr(pos_, 5) == "false") {
            result.kind = JsonValue::Kind::kBool;
            pos_ += 5;
        } else if (text_.substr(pos_, 4) == "null") {
            pos_ += 4;
        } else {
            fail("unexpected character");
        }
        return result;
    }

    std::string_view text_;
    std::size_t pos_ = 0;
};

/** Convert a JSON number to a non-negative integer, rejecting fractions.
 *
 * @param value JSON number.
 * @param what Name used in the error message.
 * @return The integer value.
 * @throws std::invalid_argument If @p value is not a non-negative integer.
 */
inline std::size_t json_size(const JsonValue& value, const std::string& what) {
    // Every integer up to 2^53 is exact in a double.
    constexpr double kLargestExact = 9007199254740992.0;
    if (value.kind != JsonValue::Kind::kNumber || value.number < 0.0 ||
        value.number > kLargestExact ||
        value.number !=
            static_cast<double>(static_cast<std::uint64_t>(value.number))) {
        throw std::invalid_argument(what + " must be a non-negative integer");
    }
    return static_cast<std::size_t>(value.number);
}

/** Read a whole file as bytes.
 *
 * @param path File to read.
 * @return The file contents.
 * @throws std::runtime_error If the file cannot be read.
 */
inline std::string read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open " + path.string());
    }
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

} // namespace detail

/** Parse a JSON document.
 *
 * @param text Complete JSON text.
 * @return The top-level value.
 * @throws std::invalid_argument If the text is not valid JSON.
 */
inline JsonValue parse_json(std::string_view text) {
    return detail::JsonParser(text).parse();
}

/** Read every F32 tensor in a safetensors file.
 *
 * @param path File to read.
 * @return Tensors by name; entries of other dtypes are skipped.
 * @throws std::runtime_error If the file cannot be read.
 * @throws std::invalid_argument If the file is not valid safetensors.
 */
inline std::map<std::string, Tensor<float>>
read_safetensors(const std::filesystem::path& path) {
    constexpr std::size_t kLengthBytes = 8;
    const std::string bytes = detail::read_file(path);
    if (bytes.size() < kLengthBytes) {
        throw std::invalid_argument(path.string() + " is too short");
    }
    std::uint64_t header_length = 0;
    std::memcpy(&header_length, bytes.data(), kLengthBytes);
    if (header_length > bytes.size() - kLengthBytes) {
        throw std::invalid_argument(path.string() + " has a truncated header");
    }
    const std::size_t data_start =
        kLengthBytes + static_cast<std::size_t>(header_length);
    const JsonValue header = parse_json(std::string_view(bytes).substr(
        kLengthBytes, data_start - kLengthBytes));
    if (header.kind != JsonValue::Kind::kObject) {
        throw std::invalid_argument("safetensors header must be an object");
    }

    std::map<std::string, Tensor<float>> tensors;
    for (std::size_t index = 0; index < header.keys.size(); ++index) {
        const std::string& name = header.keys[index];
        const JsonValue& entry = header.items[index];
        if (name == "__metadata__" || entry.at("dtype").text != "F32") {
            continue;
        }
        Tensor<float> tensor;
        for (const JsonValue& dim : entry.at("shape").items) {
            tensor.shape.push_back(detail::json_size(dim, name + " shape"));
        }
        const JsonValue& offsets = entry.at("data_offsets");
        if (offsets.items.size() != 2) {
            throw std::invalid_argument(name + " needs two data offsets");
        }
        const std::size_t begin =
            detail::json_size(offsets.items[0], name + " offset");
        const std::size_t end =
            detail::json_size(offsets.items[1], name + " offset");
        const std::size_t count = detail::element_count(tensor.shape);
        if (end < begin || end - begin != count * sizeof(float) ||
            end > bytes.size() - data_start) {
            throw std::invalid_argument(name + " has an invalid byte range");
        }
        tensor.values.resize(count);
        std::memcpy(tensor.values.data(), bytes.data() + data_start + begin,
                    end - begin);
        tensors.emplace(name, std::move(tensor));
    }
    return tensors;
}

/** Write tensors as an F32 safetensors file.
 *
 * @param path File to create or replace.
 * @param tensors Tensors by name; each value count must match its shape.
 * @throws std::invalid_argument If a tensor's values do not match its shape.
 * @throws std::runtime_error If the file cannot be written.
 */
inline void
write_safetensors(const std::filesystem::path& path,
                  const std::map<std::string, Tensor<float>>& tensors) {
    constexpr std::size_t kAlignment = 8;
    std::ostringstream header;
    header << '{';
    std::size_t offset = 0;
    bool first = true;
    for (const auto& [name, tensor] : tensors) {
        if (detail::element_count(tensor.shape) != tensor.values.size()) {
            throw std::invalid_argument(name + " values do not match shape");
        }
        header << (first ? "" : ",") << '"' << name
               << R"(":{"dtype":"F32","shape":[)";
        for (std::size_t dim = 0; dim < tensor.shape.size(); ++dim) {
            header << (dim == 0 ? "" : ",") << tensor.shape[dim];
        }
        const std::size_t size = tensor.values.size() * sizeof(float);
        header << R"(],"data_offsets":[)" << offset << ',' << offset + size
               << "]}";
        offset += size;
        first = false;
    }
    header << '}';
    std::string text = header.str();
    // Pad with spaces so the data starts 8-byte aligned, as the format
    // recommends.
    text.append((kAlignment - (text.size() % kAlignment)) % kAlignment, ' ');

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot create " + path.string());
    }
    const std::uint64_t length = text.size();
    std::array<char, sizeof length> length_bytes{};
    std::memcpy(length_bytes.data(), &length, sizeof length);
    file.write(length_bytes.data(), length_bytes.size());
    file.write(text.data(), static_cast<std::streamsize>(text.size()));
    for (const auto& entry : tensors) {
        const std::vector<float>& values = entry.second.values;
        std::string raw(values.size() * sizeof(float), '\0');
        std::memcpy(raw.data(), values.data(), raw.size());
        file.write(raw.data(), static_cast<std::streamsize>(raw.size()));
    }
    if (!file) {
        throw std::runtime_error("cannot write " + path.string());
    }
}

} // namespace inference_engine
