#include "mdescriptor/mtp4.hpp"

#include "mdescriptor/mtp_cinf_coeffs.hpp"
#include "mdescriptor/neighbor.hpp"
#include "descriptor_common.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <functional>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace mdescriptor {
using namespace detail;

namespace {

class JsonValue {
public:
    enum class Kind { Null, Bool, Number, String, Array, Object };

    Kind kind = Kind::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<JsonValue> array;
    std::map<std::string, JsonValue> object;

    const JsonValue& at(const std::string& key) const {
        const auto it = object.find(key);
        if (it == object.end()) {
            throw std::invalid_argument("MLIP-4 JSON is missing key '" + key + "'");
        }
        return it->second;
    }

    const JsonValue& at(std::size_t index) const {
        if (index >= array.size()) {
            throw std::invalid_argument("MLIP-4 JSON array index is out of range");
        }
        return array[index];
    }

    double as_number() const {
        if (kind != Kind::Number || !std::isfinite(number)) {
            throw std::invalid_argument("MLIP-4 JSON value is not a finite number");
        }
        return number;
    }

    int as_int() const {
        const double value = as_number();
        const double rounded = std::round(value);
        if (value != rounded || rounded < static_cast<double>(std::numeric_limits<int>::min())
            || rounded > static_cast<double>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("MLIP-4 JSON value is not an integer");
        }
        return static_cast<int>(rounded);
    }

    const std::string& as_string() const {
        if (kind != Kind::String) {
            throw std::invalid_argument("MLIP-4 JSON value is not a string");
        }
        return string;
    }

    bool as_bool() const {
        if (kind != Kind::Bool) {
            throw std::invalid_argument("MLIP-4 JSON value is not a boolean");
        }
        return boolean;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& input) : input_(input) {}

    JsonValue parse() {
        skip_space();
        JsonValue result = value();
        skip_space();
        if (pos_ != input_.size()) {
            fail("trailing characters");
        }
        return result;
    }

private:
    const std::string& input_;
    std::size_t pos_ = 0;

    [[noreturn]] void fail(const std::string& message) const {
        throw std::invalid_argument("invalid MLIP-4 JSON at byte " + std::to_string(pos_) + ": " + message);
    }

    void skip_space() {
        while (pos_ < input_.size()) {
            const char c = input_[pos_];
            if (c != ' ' && c != '\n' && c != '\r' && c != '\t') return;
            ++pos_;
        }
    }

    bool consume(char expected) {
        skip_space();
        if (pos_ >= input_.size() || input_[pos_] != expected) return false;
        ++pos_;
        return true;
    }

    void expect(char expected) {
        if (!consume(expected)) fail(std::string("expected '") + expected + "'");
    }

    JsonValue value() {
        skip_space();
        if (pos_ >= input_.size()) fail("unexpected end of input");
        switch (input_[pos_]) {
        case '{': return object_value();
        case '[': return array_value();
        case '"': return string_value();
        case 't': return literal("true", JsonValue::Kind::Bool, true);
        case 'f': return literal("false", JsonValue::Kind::Bool, false);
        case 'n': return literal("null", JsonValue::Kind::Null, false);
        default: return number_value();
        }
    }

    JsonValue literal(const char* text, JsonValue::Kind kind, bool boolean) {
        const std::size_t length = std::char_traits<char>::length(text);
        if (input_.compare(pos_, length, text) != 0) fail("invalid literal");
        pos_ += length;
        JsonValue result;
        result.kind = kind;
        result.boolean = boolean;
        return result;
    }

    JsonValue number_value() {
        skip_space();
        const std::size_t begin = pos_;
        if (pos_ < input_.size() && (input_[pos_] == '-' || input_[pos_] == '+')) ++pos_;
        while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) ++pos_;
        if (pos_ < input_.size() && input_[pos_] == '.') {
            ++pos_;
            while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) ++pos_;
        }
        if (pos_ < input_.size() && (input_[pos_] == 'e' || input_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < input_.size() && (input_[pos_] == '-' || input_[pos_] == '+')) ++pos_;
            while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) ++pos_;
        }
        if (begin == pos_) fail("expected a number");
        char* end = nullptr;
        const std::string token = input_.substr(begin, pos_ - begin);
        const double parsed = std::strtod(token.c_str(), &end);
        if (end == token.c_str() || *end != '\0' || !std::isfinite(parsed)) fail("invalid number");
        JsonValue result;
        result.kind = JsonValue::Kind::Number;
        result.number = parsed;
        return result;
    }

    JsonValue string_value() {
        expect('"');
        JsonValue result;
        result.kind = JsonValue::Kind::String;
        while (pos_ < input_.size()) {
            const char c = input_[pos_++];
            if (c == '"') return result;
            if (c != '\\') {
                result.string.push_back(c);
                continue;
            }
            if (pos_ >= input_.size()) fail("unterminated string escape");
            const char escaped = input_[pos_++];
            switch (escaped) {
            case '"': result.string.push_back('"'); break;
            case '\\': result.string.push_back('\\'); break;
            case '/': result.string.push_back('/'); break;
            case 'b': result.string.push_back('\b'); break;
            case 'f': result.string.push_back('\f'); break;
            case 'n': result.string.push_back('\n'); break;
            case 'r': result.string.push_back('\r'); break;
            case 't': result.string.push_back('\t'); break;
            case 'u':
                // MLIP-4 JSON keys and class names are ASCII.  Consume a
                // unicode escape while retaining a safe placeholder for the
                // uncommon non-ASCII case.
                if (pos_ + 4 > input_.size()) fail("short unicode escape");
                pos_ += 4;
                result.string.push_back('?');
                break;
            default: fail("invalid string escape");
            }
        }
        fail("unterminated string");
    }

    JsonValue array_value() {
        expect('[');
        JsonValue result;
        result.kind = JsonValue::Kind::Array;
        skip_space();
        if (consume(']')) return result;
        while (true) {
            result.array.push_back(value());
            if (consume(']')) return result;
            expect(',');
        }
    }

    JsonValue object_value() {
        expect('{');
        JsonValue result;
        result.kind = JsonValue::Kind::Object;
        skip_space();
        if (consume('}')) return result;
        while (true) {
            const JsonValue key = string_value();
            expect(':');
            result.object.emplace(key.string, value());
            if (consume('}')) return result;
            expect(',');
        }
    }
};

struct BasisKey {
    int size = 0;
    std::vector<int> elem;

    bool operator<(const BasisKey& other) const noexcept {
        if (size != other.size) return size < other.size;
        return elem < other.elem;
    }

    bool operator==(const BasisKey& other) const noexcept {
        return size == other.size && elem == other.elem;
    }

    int vector(int i) const noexcept { return elem[static_cast<std::size_t>(i)]; }
    int matrix(int i, int j) const noexcept {
        return elem[static_cast<std::size_t>(size + i * size + j)];
    }

    int dimension() const noexcept {
        int result = 0;
        for (int i = 0; i < size; ++i) {
            result += matrix(i, i);
            for (int j = i + 1; j < size; ++j) result -= 2 * matrix(i, j);
        }
        return result;
    }
};

struct EvalNode {
    enum class Kind { Input, Linear, Product };
    Kind kind = Kind::Input;
    int input = -1;
    std::vector<std::pair<int, double>> linear;
    std::vector<std::array<int, 3>> products;
};

struct Signature {
    BasisKey basis;
    std::vector<int> index;
};

struct SubBasis {
    BasisKey first;
    BasisKey second;
    std::vector<int> perm;
};

BasisKey from_json_basis(const JsonValue& value) {
    if (value.kind != JsonValue::Kind::Array) {
        throw std::invalid_argument("MLIP-4 mtp_basis entry must be an integer array");
    }
    BasisKey result;
    result.elem.reserve(value.array.size());
    for (const auto& element : value.array) result.elem.push_back(element.as_int());
    const double discriminant = 4.0 * result.elem.size() + 1.0;
    result.size = static_cast<int>(std::round(0.5 * (std::sqrt(discriminant) - 1.0)));
    if (result.size <= 0 || result.size + result.size * result.size != static_cast<int>(result.elem.size())) {
        throw std::invalid_argument("MLIP-4 basis function has a nonsquare tensor matrix");
    }
    for (int value_i : result.elem) {
        if (value_i < 0) throw std::invalid_argument("MLIP-4 basis function contains a negative index");
    }
    return result;
}

std::vector<int> canonicalize(BasisKey& basis) {
    std::vector<int> permutation(static_cast<std::size_t>(basis.size));
    std::iota(permutation.begin(), permutation.end(), 0);
    std::vector<int> best = permutation;
    const auto less = [&](const std::vector<int>& left, const std::vector<int>& right) {
        for (int i = 0; i < basis.size; ++i) {
            if (basis.vector(left[i]) != basis.vector(right[i])) {
                return basis.vector(left[i]) < basis.vector(right[i]);
            }
        }
        for (int i = 0; i < basis.size; ++i) {
            for (int j = 0; j < basis.size; ++j) {
                const int lv = basis.matrix(left[i], left[j]);
                const int rv = basis.matrix(right[i], right[j]);
                if (lv != rv) return lv < rv;
            }
        }
        return false;
    };
    do {
        if (less(permutation, best)) best = permutation;
    } while (std::next_permutation(permutation.begin(), permutation.end()));

    BasisKey old = basis;
    for (int i = 0; i < basis.size; ++i) basis.elem[static_cast<std::size_t>(i)] = old.vector(best[i]);
    for (int i = 0; i < basis.size; ++i) {
        for (int j = 0; j < basis.size; ++j) {
            basis.elem[static_cast<std::size_t>(basis.size + i * basis.size + j)] = old.matrix(best[i], best[j]);
        }
    }
    return best;
}

SubBasis construct_sub(const BasisKey& basis, const std::vector<int>& subset) {
    const int first_size = static_cast<int>(std::count(subset.begin(), subset.end(), 0));
    const int second_size = basis.size - first_size;
    SubBasis result;
    result.first.size = first_size;
    result.first.elem.assign(static_cast<std::size_t>(first_size + first_size * first_size), 0);
    result.second.size = second_size;
    result.second.elem.assign(static_cast<std::size_t>(second_size + second_size * second_size), 0);
    int sub_i[2] = {0, 0};
    for (int i = 0; i < basis.size; ++i) {
        const int which = subset[static_cast<std::size_t>(i)];
        const int index = sub_i[which]++;
        if (which == 0) result.first.elem[static_cast<std::size_t>(index)] = basis.vector(i);
        else result.second.elem[static_cast<std::size_t>(index)] = basis.vector(i);
    }
    sub_i[0] = sub_i[1] = 0;
    for (int i = 0; i < basis.size; ++i) {
        const int wi = subset[static_cast<std::size_t>(i)];
        const int ii = sub_i[wi]++;
        int sub_j[2] = {0, 0};
        for (int j = 0; j < basis.size; ++j) {
            const int wj = subset[static_cast<std::size_t>(j)];
            const int jj = sub_j[wj]++;
            if (wi == wj) {
                if (wi == 0) result.first.elem[static_cast<std::size_t>(first_size + ii * first_size + jj)] = basis.matrix(i, j);
                else result.second.elem[static_cast<std::size_t>(second_size + ii * second_size + jj)] = basis.matrix(i, j);
            }
        }
    }

    std::vector<int> temp_perm;
    temp_perm.reserve(static_cast<std::size_t>(basis.size));
    for (int which = 0; which < 2; ++which) {
        for (int i = 0; i < basis.size; ++i) if (subset[static_cast<std::size_t>(i)] == which) temp_perm.push_back(i);
    }
    const std::vector<int> first_perm = canonicalize(result.first);
    const std::vector<int> second_perm = canonicalize(result.second);
    result.perm.resize(static_cast<std::size_t>(basis.size));
    for (int i = 0; i < first_size; ++i) result.perm[static_cast<std::size_t>(i)] = temp_perm[static_cast<std::size_t>(first_perm[i])];
    for (int i = 0; i < second_size; ++i) result.perm[static_cast<std::size_t>(first_size + i)] = temp_perm[static_cast<std::size_t>(first_size + second_perm[i])];
    return result;
}

struct RadialBasis {
    std::string type;
    int size = 0;
    double mindist = 0.0;
    double maxdist = 0.0;
    double maxdist_sq = 0.0;
    double maxdist_sq_minus_eps = 0.0;
    double exp_ratio = 0.0;
    double zeroth = 0.0;
    std::vector<std::array<double, 3>> recursive;
    std::vector<double> vdw_damped_params;

    static constexpr double vdw_zeroth = 102.295067549833082;
    static constexpr double vdw_recursive[22][3] = {
        {0.0, 0.0, 0.0},
        {8.87248718788328392, -0.542267396590896327, 0.0},
        {6.79251268298387606, -0.509113772340205375, -0.112707967768682771},
        {5.97200240050575452, -0.490871142415655530, -0.147220924961263526},
        {5.53059832007591211, -0.480355753378264948, -0.167448023784336122},
        {5.25531942342471711, -0.474098576668924181, -0.180812263362180705},
        {5.06759630263062248, -0.470320600452479026, -0.190283390871098224},
        {4.93157096163622518, -0.468051693398248887, -0.197332214383551713},
        {4.82855147527615044, -0.466734005666727749, -0.202775141588597198},
        {4.74785129937755951, -0.466032246548252509, -0.207101447529418508},
        {4.68292754573870735, -0.465737360308779940, -0.210621592157087861},
        {4.62955581831805592, -0.465714968307180773, -0.213541633995589659},
        {4.58489017048410400, -0.465876479156477367, -0.216003443795458052},
        {4.54694521651720235, -0.466162247521016685, -0.218107732751734198},
        {4.51429486932813924, -0.466531414516049284, -0.219927875173733517},
        {4.48588903210495412, -0.466955593424251038, -0.221518538098693055},
        {4.46093775571628900, -0.467414841652198140, -0.222921252140461663},
        {4.43883562185532690, -0.467895031561814091, -0.224168113244483244},
        {4.41911098018499224, -0.468386099282248837, -0.225284305432789144},
        {4.40139102397022533, -0.468880857197526929, -0.226289858861643283},
        {4.38537723625471118, -0.469374175729228661, -0.227200899568782515},
        {4.37082779110205071, -0.469862411506819840, -0.228030553844448804}
    };

    void load(const std::string& new_type, const JsonValue& value) {
        type = new_type;
        size = value.at("basis_size").as_int();
        if (type == "RadialBasisVdwDamped") {
            maxdist = value.at("cutoff").as_number();
            mindist = 0.0;
            if (value.object.count("params")) {
                for (const auto& item : value.at("params").array) vdw_damped_params.push_back(item.as_number());
            } else {
                vdw_damped_params = {-0.33528, 2.86229, 1.001, 1.452};
            }
            if (size <= 0 || size > 2 || maxdist <= 0.0 || vdw_damped_params.size() < 2) {
                throw std::invalid_argument("invalid MLIP-4 RadialBasisVdwDamped parameters");
            }
            return;
        }
        mindist = value.at("mindist").as_number();
        maxdist = value.at("maxdist").as_number();
        if (size <= 0 || size > static_cast<int>(kMlip4CinfMaxSize) || maxdist <= mindist || mindist < 0.0) {
            throw std::invalid_argument("invalid MLIP-4 radial basis parameters");
        }
        maxdist_sq = maxdist * maxdist;
        recursive.assign(static_cast<std::size_t>(size), {0.0, 0.0, 0.0});
        if (type == "RadialBasisCinf") {
            const double ratio = mindist / maxdist;
            maxdist_sq_minus_eps = maxdist_sq * (49.0 + ratio) / 50.0;
            exp_ratio = -2.0 * (1.0 - ratio * ratio);
            const double angle = std::acos(1.25 * ratio);
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < 3; ++j) {
                    double accumulator = 0.0;
                    for (int k = static_cast<int>(kMlip4CinfChebSize) - 1; k >= 0; --k) {
                        accumulator += kMlip4CinfRecursive[i][j][k] * std::cos(2.0 * k * angle);
                    }
                    recursive[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = accumulator;
                }
            }
            for (int k = static_cast<int>(kMlip4CinfChebSize) - 1; k >= 0; --k) {
                zeroth += kMlip4CinfZeroth[k] * std::cos(2.0 * k * angle);
            }
        } else if (type == "RadialBasisVdw") {
            if (mindist <= 0.0) throw std::invalid_argument("RadialBasisVdw requires mindist > 0");
            for (int i = 0; i < size; ++i) recursive[static_cast<std::size_t>(i)] = {vdw_recursive[i][0], vdw_recursive[i][1], vdw_recursive[i][2]};
            maxdist_sq_minus_eps = mindist * mindist * 1.02;
        } else if (type != "RadialBasisChebyshev") {
            throw std::invalid_argument("unsupported native MLIP-4 radial basis JSON class");
        }
    }

    void calc(double r_sq, int self_species, int neighbor_species, double* output) const {
        if (type == "RadialBasisVdwDamped") {
            const double r_sq_3 = r_sq * r_sq * r_sq;
            const std::size_t self = static_cast<std::size_t>(self_species);
            const std::size_t neighbor = static_cast<std::size_t>(neighbor_species);
            if (self + 2 >= vdw_damped_params.size() || neighbor + 2 >= vdw_damped_params.size()) {
                throw std::invalid_argument("RadialBasisVdwDamped params do not cover MLIP-4 species");
            }
            const double radius_sum = vdw_damped_params[2 + self] + vdw_damped_params[2 + neighbor];
            const double damp = vdw_damped_params[0] * radius_sum + vdw_damped_params[1];
            const double damp_sq = damp * damp;
            const double damp_6 = damp_sq * damp_sq * damp_sq;
            output[0] = 100.0 / (r_sq_3 + damp_6);
            if (size > 1) {
                const double damp_8 = damp_6 * damp_sq;
                output[1] = 100.0 / (r_sq_3 * r_sq + damp_8);
            }
            return;
        }
        if (type == "RadialBasisVdw") {
            const double mindist_sq = mindist * mindist;
            if (r_sq <= mindist_sq * 1.02) {
                std::fill(output, output + size, 0.0);
                output[0] = 2.5;
                return;
            }
            if (r_sq >= maxdist_sq) {
                std::fill(output, output + size, 0.0);
                return;
            }
            const double x_sq = mindist_sq / r_sq;
            const double my_exp = std::exp(1.0 / (x_sq - 1.0));
            const double mult = x_sq * x_sq * x_sq * my_exp;
            output[0] = 2.5 * std::pow(1.0 - 2.71828182845904524 * my_exp, 3.0);
            if (size == 1) return;
            output[1] = vdw_zeroth * mult;
            double previous = 0.0;
            for (int i = 1; i < size - 1; ++i) {
                output[i + 1] = recursive[static_cast<std::size_t>(i)][0]
                    * ((x_sq + recursive[static_cast<std::size_t>(i)][1]) * output[i]
                        + recursive[static_cast<std::size_t>(i)][2] * previous);
                previous = output[i];
            }
            return;
        }
        if (type == "RadialBasisChebyshev") {
            if (r_sq >= maxdist_sq) {
                std::fill(output, output + size, 0.0);
                return;
            }
            const double r = std::sqrt(r_sq);
            const double ksi = (2.0 * r - (mindist + maxdist)) / (maxdist - mindist);
            const double edge = r - maxdist;
            output[0] = edge * edge;
            if (size > 1) output[1] = ksi * edge * edge;
            for (int i = 2; i < size; ++i) output[i] = 2.0 * ksi * output[i - 1] - output[i - 2];
            return;
        }
        if (r_sq >= maxdist_sq_minus_eps) {
            std::fill(output, output + size, 0.0);
            return;
        }
        const double x_sq = r_sq / maxdist_sq;
        const double mult = std::exp(exp_ratio / (1.0 - x_sq));
        output[0] = zeroth * mult;
        double previous = 0.0;
        for (int i = 0; i < size - 1; ++i) {
            output[i + 1] = recursive[static_cast<std::size_t>(i)][0]
                * ((x_sq + recursive[static_cast<std::size_t>(i)][1]) * output[i]
                    + recursive[static_cast<std::size_t>(i)][2] * previous);
            previous = output[i];
        }
    }
};

// The native MLIP-4 scalar-product code uses four-component expansion
// arithmetic for the LDLT dependency test.  A double here is not sufficient: near-dependent MTP
// elements can differ from zero by much less than machine epsilon after the
// Schur updates.  This compact expansion arithmetic keeps four non-overlap
// doubles and is used only while constructing the orthogonalization matrix.
inline void mtp4_quick_two_sum(double a, double b, double& s, double& e) {
    s = a + b;
    e = b - (s - a);
}

inline void mtp4_two_sum(double a, double b, double& s, double& e) {
    s = a + b;
    const double v = s - a;
    e = (a - (s - v)) + (b - v);
}

inline void mtp4_split(double a, double& hi, double& lo) {
    constexpr double factor = 134217729.0; // 2^27 + 1
    const double t = factor * a;
    hi = t - (t - a);
    lo = a - hi;
}

inline void mtp4_two_prod(double a, double b, double& p, double& e) {
    double a_hi, a_lo, b_hi, b_lo;
    mtp4_split(a, a_hi, a_lo);
    mtp4_split(b, b_hi, b_lo);
    p = a * b;
    e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
}

inline void mtp4_renormalize(double* a) {
    double s;
    double t[5];
    mtp4_quick_two_sum(a[3], a[4], s, t[4]);
    mtp4_quick_two_sum(a[2], s, s, t[3]);
    mtp4_quick_two_sum(a[1], s, s, t[2]);
    mtp4_quick_two_sum(a[0], s, t[0], t[1]);
    s = t[0];
    int k = 0;
    for (int i = 1; i <= 4; ++i) {
        double e;
        mtp4_quick_two_sum(s, t[i], s, e);
        if (e != 0.0) {
            a[k++] = s;
            s = e;
        }
    }
    a[k++] = s;
    for (; k < 4; ++k) a[k] = 0.0;
}

inline void mtp4_three_sum(double& x, double& y, double& z) {
    mtp4_two_sum(x, y, x, y);
    mtp4_two_sum(x, z, x, z);
    mtp4_two_sum(y, z, y, z);
}

inline void mtp4_three_sum_two_out(double& x, double& y, double z) {
    mtp4_two_sum(x, y, x, y);
    mtp4_two_sum(x, z, x, z);
    y += z;
}

inline void mtp4_six_sum_three_out(double& x, double& y, double& z, double f, double g, double h) {
    mtp4_three_sum(x, y, z);
    mtp4_three_sum(f, g, h);
    mtp4_two_sum(x, f, x, f);
    mtp4_two_sum(y, g, y, g);
    mtp4_two_sum(y, f, y, f);
    z += h;
    z += g;
    z += f;
}

inline void mtp4_double_double_add(double& x, double& y, double a, double b) {
    mtp4_two_sum(x, a, x, a);
    y += b;
    y += a;
    mtp4_quick_two_sum(x, y, x, y);
}

inline void mtp4_nine_sum_two_out(double& x0, double& x1, double x2,
                                  double x3, double x4, double x5,
                                  double x6, double x7, double x8) {
    mtp4_two_sum(x0, x1, x0, x1);
    mtp4_two_sum(x2, x3, x2, x3);
    mtp4_double_double_add(x0, x1, x2, x3);
    mtp4_two_sum(x4, x5, x4, x5);
    mtp4_two_sum(x6, x7, x6, x7);
    mtp4_double_double_add(x4, x5, x6, x7);
    mtp4_double_double_add(x0, x1, x4, x5);
    mtp4_three_sum_two_out(x0, x1, x8);
}

class Mtp4Quad {
public:
    double a[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

    Mtp4Quad() = default;
    Mtp4Quad(double value) { a[0] = value; }

    explicit operator double() const { return a[0]; }

    Mtp4Quad& operator+=(double value) {
        mtp4_two_sum(a[0], value, a[0], value);
        mtp4_two_sum(a[1], value, a[1], value);
        mtp4_two_sum(a[2], value, a[2], value);
        mtp4_two_sum(a[3], value, a[3], a[4]);
        mtp4_renormalize(a);
        return *this;
    }
    Mtp4Quad& operator+=(Mtp4Quad value) {
        mtp4_two_sum(a[0], value.a[0], a[0], value.a[0]);
        mtp4_two_sum(a[1], value.a[1], a[1], value.a[1]);
        mtp4_two_sum(a[1], value.a[0], a[1], value.a[0]);
        mtp4_two_sum(a[2], value.a[2], a[2], value.a[2]);
        mtp4_three_sum(a[2], value.a[0], value.a[1]);
        mtp4_two_sum(a[3], value.a[3], a[3], value.a[3]);
        mtp4_three_sum_two_out(a[3], value.a[0], value.a[2]);
        a[4] = value.a[0] + value.a[1] + value.a[3];
        mtp4_renormalize(a);
        return *this;
    }
    Mtp4Quad& operator-=(double value) { return (*this += -value); }
    Mtp4Quad& operator-=(Mtp4Quad value) { return (*this += -value); }
    Mtp4Quad& operator*=(double value) {
        double e0, e1, e3;
        mtp4_two_prod(a[0], value, a[0], e0);
        mtp4_two_prod(a[1], value, a[1], e1);
        mtp4_two_sum(a[1], e0, a[1], e0);
        mtp4_two_prod(a[2], value, a[2], e3);
        mtp4_three_sum(a[2], e0, e1);
        a[3] *= value;
        mtp4_three_sum_two_out(a[3], e0, e3);
        a[4] = e0 + e1;
        mtp4_renormalize(a);
        return *this;
    }
    Mtp4Quad& operator*=(Mtp4Quad value) {
        double e0, e10;
        mtp4_two_prod(a[0], value.a[0], e0, e10);
        double e11, e12, e20, e21;
        mtp4_two_prod(a[1], value.a[0], e11, e20);
        mtp4_two_prod(a[0], value.a[1], e12, e21);
        mtp4_three_sum(e10, e11, e12);
        double& e22 = e11;
        double& e30 = e12;
        double e23, e24, e25, e31, e32, e33;
        mtp4_two_prod(a[0], value.a[2], e23, e31);
        mtp4_two_prod(a[1], value.a[1], e24, e32);
        mtp4_two_prod(a[2], value.a[0], e25, e33);
        mtp4_six_sum_three_out(e20, e21, e22, e23, e24, e25);
        double& e34 = e21;
        mtp4_nine_sum_two_out(e30, e31, e32, e33, e34,
                              a[0] * value.a[3], a[1] * value.a[2],
                              a[2] * value.a[1], a[3] * value.a[0]);
        a[0] = e0;
        a[1] = e10;
        a[2] = e20;
        a[3] = e30;
        a[4] = 0.0;
        mtp4_renormalize(a);
        return *this;
    }
    Mtp4Quad& operator/=(Mtp4Quad divisor) {
        Mtp4Quad remainder(*this);
        for (int k = 0; k < 4; ++k) {
            a[k] = remainder.a[0] / divisor.a[0];
            remainder -= divisor * a[k];
        }
        mtp4_renormalize(a);
        return *this;
    }

    friend Mtp4Quad operator-(Mtp4Quad value) {
        for (int i = 0; i < 4; ++i) value.a[i] = -value.a[i];
        return value;
    }
    friend Mtp4Quad operator+(Mtp4Quad a, double b) { return a += b; }
    friend Mtp4Quad operator+(double a, Mtp4Quad b) { return b += a; }
    friend Mtp4Quad operator+(Mtp4Quad a, Mtp4Quad b) { return a += b; }
    friend Mtp4Quad operator-(Mtp4Quad a, double b) { return a -= b; }
    friend Mtp4Quad operator-(double a, Mtp4Quad b) { return -(b -= a); }
    friend Mtp4Quad operator-(Mtp4Quad a, Mtp4Quad b) { return a -= b; }
    friend Mtp4Quad operator*(Mtp4Quad a, double b) { return a *= b; }
    friend Mtp4Quad operator*(double a, Mtp4Quad b) { return b *= a; }
    friend Mtp4Quad operator*(Mtp4Quad a, Mtp4Quad b) { return a *= b; }
    friend Mtp4Quad operator/(Mtp4Quad a, Mtp4Quad b) { return a /= b; }
    friend bool operator<(const Mtp4Quad& x, const Mtp4Quad& y) {
        for (int i = 0; i < 4; ++i) {
            if (x.a[i] != y.a[i]) return x.a[i] < y.a[i];
        }
        return false;
    }
    friend bool operator>(const Mtp4Quad& x, const Mtp4Quad& y) { return y < x; }
    friend Mtp4Quad abs(Mtp4Quad value) {
        if (value.a[0] < 0.0) value = -value;
        return value;
    }
};

using ScalarProductFloat = Mtp4Quad;

using PowerPolynomial = std::map<std::array<int, 3>, ScalarProductFloat>;

ScalarProductFloat scalar_double_factorial(int value) {
    ScalarProductFloat result = 1.0;
    for (int i = value; i > 0; i -= 2) result *= static_cast<double>(i);
    return result;
}

PowerPolynomial expand_trace_power(const std::array<int, 3>& index, std::map<std::array<int, 3>, PowerPolynomial>& cache) {
    const auto found = cache.find(index);
    if (found != cache.end()) return found->second;
    PowerPolynomial result;
    if (index[2] <= 1) {
        result[index] = 1.0;
    } else {
        auto first = index;
        auto second = index;
        auto third = index;
        first[2] -= 2;
        second[2] -= 2;
        second[0] += 2;
        third[2] -= 2;
        third[1] += 2;
        for (const auto& term : expand_trace_power(first, cache)) result[term.first] += term.second;
        for (const auto& term : expand_trace_power(second, cache)) result[term.first] -= term.second;
        for (const auto& term : expand_trace_power(third, cache)) result[term.first] -= term.second;
    }
    cache.emplace(index, result);
    return result;
}

ScalarProductFloat multi_integrate_sq(const std::vector<std::array<int, 3>>& indices) {
    std::map<std::array<int, 3>, PowerPolynomial> cache;
    std::vector<PowerPolynomial> expanded;
    expanded.reserve(indices.size());
    for (const auto& index : indices) expanded.push_back(expand_trace_power(index, cache));
    ScalarProductFloat result = 0.0;
    std::map<std::array<int, 3>, int> multiplicities;
    const std::function<void(std::size_t, ScalarProductFloat)> visit =
        [&](std::size_t position, ScalarProductFloat coefficient) {
            if (position == expanded.size()) {
                ScalarProductFloat contribution = coefficient;
                for (const auto& item : multiplicities) {
                    if ((item.second & 1) != 0) {
                        contribution = 0.0;
                        break;
                    }
                    contribution *= scalar_double_factorial(item.second - 1);
                }
                result += contribution;
                return;
            }
            for (const auto& term : expanded[position]) {
                ++multiplicities[term.first];
                visit(position + 1, coefficient * term.second);
                if (--multiplicities[term.first] == 0) multiplicities.erase(term.first);
            }
        };
    visit(0, ScalarProductFloat(1.0));
    return result;
}

ScalarProductFloat multi_integrate(BasisKey basis, std::vector<std::array<int, 3>> dimensions) {
    for (int i = 0; i < basis.size; ++i) {
        for (int j = i + 1; j < basis.size; ++j) {
            if (basis.matrix(i, j) <= 0) continue;
            --basis.elem[static_cast<std::size_t>(basis.size + i * basis.size + j)];
            --basis.elem[static_cast<std::size_t>(basis.size + j * basis.size + i)];
            ScalarProductFloat result = 0.0;
            for (int component = 0; component < 3; ++component) {
                ++dimensions[static_cast<std::size_t>(i)][static_cast<std::size_t>(component)];
                ++dimensions[static_cast<std::size_t>(j)][static_cast<std::size_t>(component)];
                result += multi_integrate(basis, dimensions);
                --dimensions[static_cast<std::size_t>(i)][static_cast<std::size_t>(component)];
                --dimensions[static_cast<std::size_t>(j)][static_cast<std::size_t>(component)];
            }
            return result;
        }
    }
    ScalarProductFloat result = 1.0;
    int start = 0;
    while (start < basis.size) {
        int end = start + 1;
        while (end < basis.size && basis.vector(end) == basis.vector(start)) ++end;
        std::vector<std::array<int, 3>> group;
        for (int i = start; i < end; ++i) group.push_back(dimensions[static_cast<std::size_t>(i)]);
        result *= multi_integrate_sq(group);
        start = end;
    }
    return result;
}

ScalarProductFloat scalar_product(const Signature& left, const Signature& right) {
    std::map<int, std::array<int, 2>> radial_counts;
    std::map<int, std::array<int, 2>> parity_counts;
    for (int side = 0; side < 2; ++side) {
        const BasisKey& basis = side == 0 ? left.basis : right.basis;
        for (int i = 0; i < basis.size; ++i) {
            ++radial_counts[basis.vector(i)][static_cast<std::size_t>(side)];
            ++parity_counts[basis.vector(i)][static_cast<std::size_t>(basis.matrix(i, i) & 1)];
        }
    }
    for (const auto& item : radial_counts) {
        if (item.second[0] != item.second[1]) return 0.0;
    }
    for (const auto& item : parity_counts) {
        if ((item.second[0] & 1) != 0 || (item.second[1] & 1) != 0) return 0.0;
    }

    // MLIP-4 treats radial-channel names as dummy labels in this angular
    // scalar product.  Relabel them by multiplicity and tensor rank, exactly
    // as MTP_BasisScalarProduct::scalar_product does.
    BasisKey left_basis = left.basis;
    BasisKey right_basis = right.basis;
    int radial_count = 0;
    for (int side = 0; side < 2; ++side) {
        const BasisKey& basis = side == 0 ? left_basis : right_basis;
        for (int i = 0; i < basis.size; ++i) radial_count = std::max(radial_count, basis.vector(i) + 1);
    }
    std::vector<std::array<int, 3>> radial_signature(static_cast<std::size_t>(radial_count));
    for (int i = 0; i < radial_count; ++i) radial_signature[static_cast<std::size_t>(i)][2] = i;
    for (int side = 0; side < 2; ++side) {
        const BasisKey& basis = side == 0 ? left_basis : right_basis;
        for (int i = 0; i < basis.size; ++i) {
            auto& signature = radial_signature[static_cast<std::size_t>(basis.vector(i))];
            --signature[0];
            signature[1] -= basis.matrix(i, i);
        }
    }
    std::stable_sort(radial_signature.begin(), radial_signature.end());
    std::vector<int> radial_remap(static_cast<std::size_t>(radial_count));
    for (int i = 0; i < radial_count; ++i) radial_remap[static_cast<std::size_t>(radial_signature[static_cast<std::size_t>(i)][2])] = i;
    for (int side = 0; side < 2; ++side) {
        BasisKey& basis = side == 0 ? left_basis : right_basis;
        for (int i = 0; i < basis.size; ++i) basis.elem[static_cast<std::size_t>(i)] = radial_remap[static_cast<std::size_t>(basis.vector(i))];
    }

    BasisKey combined;
    combined.size = left_basis.size + right_basis.size;
    combined.elem.assign(static_cast<std::size_t>(combined.size + combined.size * combined.size), 0);
    for (int i = 0; i < left_basis.size; ++i) combined.elem[static_cast<std::size_t>(i)] = left_basis.vector(i);
    for (int i = 0; i < right_basis.size; ++i) combined.elem[static_cast<std::size_t>(left_basis.size + i)] = right_basis.vector(i);
    for (int i = 0; i < left_basis.size; ++i) for (int j = 0; j < left_basis.size; ++j)
        combined.elem[static_cast<std::size_t>(combined.size + i * combined.size + j)] = left_basis.matrix(i, j);
    for (int i = 0; i < right_basis.size; ++i) for (int j = 0; j < right_basis.size; ++j)
        combined.elem[static_cast<std::size_t>(combined.size + (left_basis.size + i) * combined.size + left_basis.size + j)] = right_basis.matrix(i, j);

    std::vector<int> permutation(static_cast<std::size_t>(combined.size));
    std::iota(permutation.begin(), permutation.end(), 0);
    std::stable_sort(permutation.begin(), permutation.end(), [&](int a, int b) {
        if (combined.vector(a) != combined.vector(b)) return combined.vector(a) < combined.vector(b);
        return combined.matrix(a, a) < combined.matrix(b, b);
    });
    BasisKey canonical = combined;
    for (int i = 0; i < combined.size; ++i) canonical.elem[static_cast<std::size_t>(i)] = combined.vector(permutation[i]);
    for (int i = 0; i < combined.size; ++i) for (int j = 0; j < combined.size; ++j)
        canonical.elem[static_cast<std::size_t>(combined.size + i * combined.size + j)] = combined.matrix(permutation[i], permutation[j]);
    std::vector<int> inverse(static_cast<std::size_t>(combined.size));
    for (int i = 0; i < combined.size; ++i) inverse[static_cast<std::size_t>(permutation[i])] = i;

    std::vector<std::array<int, 3>> dimensions(static_cast<std::size_t>(combined.size), {0, 0, 0});
    for (int which = 0; which < 2; ++which) {
        const BasisKey& basis = which == 0 ? left_basis : right_basis;
        const std::vector<int>& index = which == 0 ? left.index : right.index;
        const int offset = which == 0 ? 0 : left_basis.size;
        int index_id = 0;
        for (int i = 0; i < basis.size; ++i) {
            int reduced = basis.matrix(i, i);
            for (int j = 0; j < basis.size; ++j) if (j != i) reduced -= basis.matrix(i, j);
            for (int k = 0; k < reduced; ++k) {
                ++dimensions[static_cast<std::size_t>(inverse[offset + i])][static_cast<std::size_t>(index[static_cast<std::size_t>(index_id++)])];
            }
        }
    }
    return multi_integrate(canonical, dimensions);
}

} // namespace

struct NativeMtp4Model::Impl {
    struct Node {
        BasisKey key;
        Node* from[2] = {nullptr, nullptr};
        std::vector<int> subset;
        std::vector<int> perm;
        std::vector<int> element_ids;
    };

    struct ParamData {
        std::vector<double> values;
        double radial_scaling = 1.0;
    };

    int species_count = 0;
    int radial_function_count = 0;
    std::vector<std::int32_t> species_order;
    ParamData params;
    RadialBasis radial_basis;
    bool is_orth = false;
    std::map<BasisKey, std::unique_ptr<Node>> nodes;
    std::vector<BasisKey> requested_basis;
    std::vector<std::array<int, 4>> moments;
    std::map<std::array<int, 4>, int> moment_to_id;
    std::vector<EvalNode> eval;
    std::vector<Signature> signatures;
    std::vector<int> scalar_output_ids;
    std::vector<bool> eliminated;

    Node* add(const BasisKey& key) {
        auto found = nodes.find(key);
        if (found != nodes.end()) return found->second.get();
        auto node = std::make_unique<Node>();
        node->key = key;
        Node* pointer = node.get();
        nodes.emplace(key, std::move(node));
        for (int i = 0; i < key.size; ++i) radial_function_count = std::max(radial_function_count, key.vector(i) + 1);
        if (key.size == 1) return pointer;

        std::vector<int> subset(static_cast<std::size_t>(key.size), 0);
        subset.back() = 1;
        std::vector<int> best_subset;
        int best_ops = std::numeric_limits<int>::max();
        int best_found = 0;
        bool valid = true;
        while (valid) {
            int off_diag = 0;
            for (int i = 0; i < key.size; ++i) for (int j = i + 1; j < key.size; ++j)
                if (subset[static_cast<std::size_t>(i)] != subset[static_cast<std::size_t>(j)]) off_diag += key.matrix(i, j);
            const int ops = key.dimension() + off_diag;
            const SubBasis candidate = construct_sub(key, subset);
            const int found_count = static_cast<int>(nodes.count(candidate.first)) + static_cast<int>(nodes.count(candidate.second));
            if (ops < best_ops || (ops == best_ops && found_count > best_found)) {
                best_ops = ops;
                best_found = found_count;
                best_subset = subset;
            }

            int i = key.size - 1;
            int grew = 0;
            for (; i >= 0 && subset[static_cast<std::size_t>(i)] == 1; --i) {
                subset[static_cast<std::size_t>(i)] = 0;
                ++grew;
            }
            if (i < 0) {
                valid = false;
            } else {
                subset[static_cast<std::size_t>(i)] = 1;
                --grew;
                for (; i >= 0 && subset[static_cast<std::size_t>(i)] == 1; --i) {}
                if (i < 0) valid = false;
            }
        }
        const SubBasis chosen = construct_sub(key, best_subset);
        pointer->subset = best_subset;
        pointer->perm = chosen.perm;
        pointer->from[0] = add(chosen.first);
        pointer->from[1] = add(chosen.second);
        return pointer;
    }

    void process() {
        for (const auto& item : nodes) {
            Node* node = item.second.get();
            if (node->key.size != 1) break;
            const int nu = node->key.matrix(0, 0);
            const int count = static_cast<int>(std::pow(3, nu));
            node->element_ids.reserve(static_cast<std::size_t>(count));
            for (int flat = 0; flat < count; ++flat) {
                int value = flat;
                std::array<int, 4> moment{node->key.vector(0), 0, 0, 0};
                for (int k = nu - 1; k >= 0; --k) {
                    const int component = value % 3;
                    value /= 3;
                    ++moment[static_cast<std::size_t>(1 + component)];
                }
                auto where = moment_to_id.find(moment);
                int id = 0;
                if (where == moment_to_id.end()) {
                    id = static_cast<int>(moments.size());
                    moment_to_id.emplace(moment, id);
                    moments.push_back(moment);
                } else id = where->second;
                node->element_ids.push_back(id);
            }
        }
        eval.resize(moments.size());
        for (std::size_t id = 0; id < moments.size(); ++id) {
            const auto& moment = moments[id];
            if (moment[3] <= 1) {
                eval[id].kind = EvalNode::Kind::Input;
                eval[id].input = static_cast<int>(id);
            } else {
                eval[id].kind = EvalNode::Kind::Linear;
                eval[id].linear = {
                    {moment_to_id.at({moment[0], moment[1], moment[2], moment[3] - 2}), 1.0},
                    {moment_to_id.at({moment[0], moment[1] + 2, moment[2], moment[3] - 2}), -1.0},
                    {moment_to_id.at({moment[0], moment[1], moment[2] + 2, moment[3] - 2}), -1.0}};
            }
        }

        for (const auto& item : nodes) {
            Node* node = item.second.get();
            if (node->key.size == 1) continue;
            const int dim = node->key.dimension();
            const int sub_dim[2] = {node->from[0]->key.dimension(), node->from[1]->key.dimension()};
            const int codim = (sub_dim[0] + sub_dim[1] - dim) / 2;
            const int total_dim = sub_dim[0] + sub_dim[1];
            std::vector<int> inverse_perm(static_cast<std::size_t>(node->key.size));
            for (int i = 0; i < node->key.size; ++i) inverse_perm[static_cast<std::size_t>(node->perm[i])] = i;
            std::vector<int> reduced(static_cast<std::size_t>(node->key.size));
            for (int i = 0; i < node->key.size; ++i) {
                reduced[static_cast<std::size_t>(i)] = node->key.matrix(i, i);
                for (int j = 0; j < node->key.size; ++j)
                    if (i != j && node->subset[static_cast<std::size_t>(i)] == node->subset[static_cast<std::size_t>(j)]) reduced[static_cast<std::size_t>(i)] -= node->key.matrix(i, j);
            }
            std::vector<std::vector<int>> assignment(static_cast<std::size_t>(node->key.size));
            std::vector<std::vector<int>> summation(static_cast<std::size_t>(node->key.size));
            int index_count = 0;
            for (int i = 0; i < node->key.size; ++i) {
                assignment[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(reduced[static_cast<std::size_t>(i)]));
                for (int& value : assignment[static_cast<std::size_t>(i)]) value = index_count++;
            }
            const int offset = node->from[0]->key.size;
            for (int i = 0; i < node->from[0]->key.size; ++i) {
                for (int j = 0; j < node->from[1]->key.size; ++j) {
                    const int left = node->perm[i];
                    const int right = node->perm[offset + j];
                    for (int k = 0; k < node->key.matrix(left, right); ++k) {
                        assignment[static_cast<std::size_t>(left)].pop_back();
                        assignment[static_cast<std::size_t>(right)].pop_back();
                        summation[static_cast<std::size_t>(left)].push_back(index_count);
                        summation[static_cast<std::size_t>(right)].push_back(index_count++);
                    }
                }
            }
            int dim_id = 0;
            for (auto& row : assignment) for (int& value : row) if (value < total_dim) value = dim_id++;
            for (auto& row : summation) for (int& value : row) if (value < total_dim) value = dim_id++;

            std::vector<std::array<int, 2>> assignment_map(static_cast<std::size_t>(dim), {-1, 1});
            std::vector<int> sum_map[2];
            sum_map[0].assign(static_cast<std::size_t>(codim), -1);
            sum_map[1].assign(static_cast<std::size_t>(codim), -1);
            for (int which = 0; which < 2; ++which) {
                Node* sub = node->from[which];
                int sub_index = 0;
                const int sub_offset = which == 0 ? 0 : offset;
                for (int i = 0; i < sub->key.size; ++i) {
                    const int full_i = node->perm[sub_offset + i];
                    for (int value : assignment[static_cast<std::size_t>(full_i)]) assignment_map[static_cast<std::size_t>(value)] = {which, sub_index++};
                    for (int value : summation[static_cast<std::size_t>(full_i)]) sum_map[which][static_cast<std::size_t>(value - total_dim)] = sub_index++;
                }
            }

            const int element_count = static_cast<int>(std::pow(3, dim));
            node->element_ids.reserve(static_cast<std::size_t>(element_count));
            for (int flat = 0; flat < element_count; ++flat) {
                int value = flat;
                std::vector<int> mi(static_cast<std::size_t>(dim));
                for (int i = dim - 1; i >= 0; --i) {
                    mi[static_cast<std::size_t>(i)] = value % 3;
                    value /= 3;
                }
                std::vector<int> sub_mi[2];
                sub_mi[0].resize(static_cast<std::size_t>(sub_dim[0]));
                sub_mi[1].resize(static_cast<std::size_t>(sub_dim[1]));
                for (int i = 0; i < dim; ++i) {
                    const auto destination = assignment_map[static_cast<std::size_t>(i)];
                    sub_mi[destination[0]][static_cast<std::size_t>(destination[1])] = mi[static_cast<std::size_t>(i)];
                }
                EvalNode expression;
                expression.kind = EvalNode::Kind::Product;
                for (int contraction = 0; contraction < static_cast<int>(std::pow(3, codim)); ++contraction) {
                    int contraction_value = contraction;
                    for (int c = codim - 1; c >= 0; --c) {
                        const int component = contraction_value % 3;
                        contraction_value /= 3;
                        sub_mi[0][static_cast<std::size_t>(sum_map[0][static_cast<std::size_t>(c)])] = component;
                        sub_mi[1][static_cast<std::size_t>(sum_map[1][static_cast<std::size_t>(c)])] = component;
                    }
                    auto flat_index = [](const std::vector<int>& index) {
                        int result = 0;
                        for (int component : index) result = result * 3 + component;
                        return result;
                    };
                    expression.products.push_back({
                        node->from[0]->element_ids[static_cast<std::size_t>(flat_index(sub_mi[0]))],
                        node->from[1]->element_ids[static_cast<std::size_t>(flat_index(sub_mi[1]))],
                        1});
                }
                // MLIP-4 creates one aggregate element variable for every
                // tensor element.  The official Variable identity is unique
                // even when two element expressions happen to be algebraically
                // equivalent, so keep the same one-to-one element/signature
                // numbering here.
                const int expression_id = static_cast<int>(eval.size());
                eval.push_back(std::move(expression));
                signatures.push_back({node->key, mi});
                node->element_ids.push_back(expression_id);
            }
        }
        eliminated.assign(signatures.size(), false);
        if (is_orth && !signatures.empty()) {
            const std::size_t n = signatures.size();
            std::vector<ScalarProductFloat> lower(n * n, 0.0);
            std::vector<ScalarProductFloat> diagonal(n, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j <= i; ++j) {
                    ScalarProductFloat value = scalar_product(signatures[i], signatures[j]);
                    for (std::size_t k = 0; k < j; ++k) value -= lower[i * n + k] * lower[j * n + k] * diagonal[k];
                    if (i == j) diagonal[i] = value;
                    else lower[i * n + j] = abs(diagonal[j]) > 1e-30
                        ? (value / diagonal[j]) : 0.0;
                }
            }
            for (std::size_t i = 0; i < n; ++i) eliminated[i] = abs(diagonal[i]) < 1e-20;
        }
        for (const auto& key : requested_basis) {
            const auto found = nodes.find(key);
            if (found == nodes.end() || key.dimension() != 0) continue;
            const int id = found->second->element_ids.front();
            const bool is_eliminated = id >= static_cast<int>(moments.size())
                && eliminated[static_cast<std::size_t>(id - static_cast<int>(moments.size()))];
            if (!is_eliminated) scalar_output_ids.push_back(id);
        }
    }

    double eval_value(int id, const std::vector<double>& input, std::vector<double>& cache, std::vector<char>& done) const {
        if (done[static_cast<std::size_t>(id)]) return cache[static_cast<std::size_t>(id)];
        const EvalNode& node = eval[static_cast<std::size_t>(id)];
        double result = 0.0;
        if (node.kind == EvalNode::Kind::Input) result = input[static_cast<std::size_t>(node.input)];
        else if (node.kind == EvalNode::Kind::Linear) {
            for (const auto& term : node.linear) result += term.second * eval_value(term.first, input, cache, done);
        } else {
            for (const auto& term : node.products) result += term[2] * eval_value(term[0], input, cache, done) * eval_value(term[1], input, cache, done);
        }
        done[static_cast<std::size_t>(id)] = 1;
        cache[static_cast<std::size_t>(id)] = result;
        return result;
    }
};

NativeMtp4Model::NativeMtp4Model() : impl_(std::make_unique<Impl>()) {}
NativeMtp4Model::~NativeMtp4Model() = default;
NativeMtp4Model::NativeMtp4Model(NativeMtp4Model&&) noexcept = default;
NativeMtp4Model& NativeMtp4Model::operator=(NativeMtp4Model&&) noexcept = default;

void NativeMtp4Model::load(const std::string& path) {
    std::ifstream stream(path);
    if (!stream.is_open()) throw std::invalid_argument("cannot open MLIP-4 JSON potential '" + path + "'");
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    const JsonValue root = JsonParser(buffer.str()).parse();
    const JsonValue* potential = &root;
    if (root.kind == JsonValue::Kind::Array && root.array.size() == 2 && root.at(0).kind == JsonValue::Kind::String) potential = &root.at(1);
    if (potential->kind != JsonValue::Kind::Object) throw std::invalid_argument("MLIP-4 potential JSON must be an object or [class, object]");
    const JsonValue& pair = potential->object.count("PairDescriptorPot") ? potential->at("PairDescriptorPot") : potential->at("pair_descriptor_pot");
    const JsonValue& basis = potential->at("mtp_basis");

    impl_ = std::make_unique<Impl>();
    const auto& species = pair.at("species_order").array;
    impl_->species_order.reserve(species.size());
    for (const auto& value : species) impl_->species_order.push_back(static_cast<std::int32_t>(value.as_int()));
    impl_->species_count = static_cast<int>(impl_->species_order.size());
    impl_->params.values.reserve(pair.at("params").array.size());
    for (const auto& value : pair.at("params").array) impl_->params.values.push_back(value.as_number());
    if (pair.object.count("hypers") && !pair.at("hypers").array.empty()) impl_->params.radial_scaling = pair.at("hypers").at(0).as_number();
    const JsonValue& radial = pair.at("radial_basis");
    if (radial.kind != JsonValue::Kind::Array || radial.array.size() != 2) throw std::invalid_argument("invalid MLIP-4 radial_basis JSON");
    impl_->radial_basis.load(radial.at(0).as_string(), radial.at(1));
    impl_->is_orth = basis.at("is_orth").as_bool();
    for (const auto& value : basis.at("basis_functions").array) impl_->requested_basis.push_back(from_json_basis(value));
    std::sort(impl_->requested_basis.begin(), impl_->requested_basis.end());
    impl_->requested_basis.erase(std::unique(impl_->requested_basis.begin(), impl_->requested_basis.end()), impl_->requested_basis.end());
    for (const auto& value : impl_->requested_basis) impl_->add(value);
    impl_->process();
    const std::size_t radial_count = static_cast<std::size_t>(impl_->species_count) * impl_->species_count
        * static_cast<std::size_t>(impl_->radial_function_count) * impl_->radial_basis.size;
    const std::size_t required = radial_count + static_cast<std::size_t>(impl_->species_count) + impl_->scalar_output_ids.size();
    if (impl_->params.values.size() < required) throw std::invalid_argument("MLIP-4 MTP params is shorter than the parsed descriptor basis");
}

std::int64_t NativeMtp4Model::feature_count() const noexcept { return static_cast<std::int64_t>(impl_->scalar_output_ids.size()); }
int NativeMtp4Model::species_count() const noexcept { return impl_->species_count; }
double NativeMtp4Model::min_dist() const noexcept { return impl_->radial_basis.mindist; }
double NativeMtp4Model::max_dist() const noexcept { return impl_->radial_basis.maxdist; }
int NativeMtp4Model::radial_basis_size() const noexcept { return impl_->radial_basis.size; }
int NativeMtp4Model::radial_funcs_count() const noexcept { return impl_->radial_function_count; }
const std::string& NativeMtp4Model::radial_basis_type() const noexcept { return impl_->radial_basis.type; }
bool NativeMtp4Model::orthogonalized() const noexcept { return impl_->is_orth; }

void NativeMtp4Model::compute(
    const StructureBatchView& batch,
    const std::vector<std::int32_t>& species,
    int num_threads,
    double* output,
    const std::shared_ptr<ComputeControl>& control
) const {
    validate_common(batch);
    if (static_cast<int>(species.size()) != impl_->species_count) throw std::invalid_argument("MLIP-4 potential species_order does not match calculator species");
    const auto graph = build_neighbor_graph(batch, impl_->radial_basis.maxdist, control, num_threads);
    const auto mapping = species_map(species);
    const std::size_t feature_count = impl_->scalar_output_ids.size();
    run_parallel_structures(batch.structures, num_threads, control, [&](std::int64_t structure) {
        for (std::int64_t center = batch.offsets[structure]; center < batch.offsets[structure + 1]; ++center) {
            if (cancelled(control)) return;
            std::vector<double> raw(impl_->eval.size(), 0.0);
            const auto central_it = mapping.find(batch.numbers[center]);
            if (central_it == mapping.end()) throw std::invalid_argument("batch contains an atomic number outside MLIP-4 species_order");
            const int central = central_it->second;
            const auto neighbors = graph.for_center(center);
            std::vector<double> radial_values(static_cast<std::size_t>(impl_->radial_function_count));
            std::vector<double> radial_basis(static_cast<std::size_t>(impl_->radial_basis.size));
            for (std::size_t n = 0; n < neighbors.size; ++n) {
                if (neighbors.exact_self(n, center)) continue;
                const auto atom = neighbors.atoms[n];
                const auto outer_it = mapping.find(batch.numbers[atom]);
                if (outer_it == mapping.end()) continue;
                const double r_sq = neighbors.distance2[n];
                const double distance = std::sqrt(std::max(r_sq, 0.0));
                if (distance <= 0.0 || distance > impl_->radial_basis.maxdist) continue;
                impl_->radial_basis.calc(r_sq, central, outer_it->second, radial_basis.data());
                const std::size_t pair_offset = (static_cast<std::size_t>(central) * impl_->species_count + static_cast<std::size_t>(outer_it->second))
                    * static_cast<std::size_t>(impl_->radial_function_count) * impl_->radial_basis.size;
                for (int rf = 0; rf < impl_->radial_function_count; ++rf) {
                    double value = 0.0;
                    for (int rb = 0; rb < impl_->radial_basis.size; ++rb) {
                        value += impl_->params.values[pair_offset + static_cast<std::size_t>(rf * impl_->radial_basis.size + rb)]
                            * radial_basis[static_cast<std::size_t>(rb)] * impl_->params.radial_scaling;
                    }
                    radial_values[static_cast<std::size_t>(rf)] = value;
                }
                const double inv_distance = 1.0 / distance;
                const double ux = neighbors.displacements[n * 3] * inv_distance;
                const double uy = neighbors.displacements[n * 3 + 1] * inv_distance;
                const double uz = neighbors.displacements[n * 3 + 2] * inv_distance;
                for (std::size_t id = 0; id < impl_->moments.size(); ++id) {
                    const auto& moment = impl_->moments[id];
                    if (moment[3] > 1) continue;
                    double angular = 1.0;
                    for (int k = 0; k < moment[1]; ++k) angular *= ux;
                    for (int k = 0; k < moment[2]; ++k) angular *= uy;
                    for (int k = 0; k < moment[3]; ++k) angular *= uz;
                    raw[id] += radial_values[static_cast<std::size_t>(moment[0])] * angular;
                }
            }
            std::vector<double> cache(impl_->eval.size(), 0.0);
            std::vector<char> done(impl_->eval.size(), 0);
            for (std::size_t id = 0; id < impl_->scalar_output_ids.size(); ++id) {
                output[center * static_cast<std::int64_t>(feature_count) + static_cast<std::int64_t>(id)]
                    = impl_->eval_value(impl_->scalar_output_ids[id], raw, cache, done);
            }
        }
    });
}

} // namespace mdescriptor
