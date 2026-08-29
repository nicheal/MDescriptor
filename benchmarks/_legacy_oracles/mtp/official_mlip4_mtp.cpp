#include "pch.h"

#include <configuration.h>
#include <pots/mtp/mtp.h>
#include <pots/mtp/mtp_levels.h>
#include <pots/radial_basis/cinf/rb_cinf.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>

namespace {

using Clock = std::chrono::steady_clock;

std::vector<Cfg> read_configurations(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open configuration input: " + path);

    std::vector<Cfg> configurations;
    std::string line;
    while (std::getline(input, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        std::stringstream stream(line);
        json_io::Reader reader(stream);
        Cfg configuration;
        if ((reader >> configuration).is_error()) {
            throw std::runtime_error("cannot parse configuration line in: " + path);
        }
        configurations.push_back(std::move(configuration));
    }
    if (configurations.empty()) {
        throw std::runtime_error("configuration input is empty: " + path);
    }
    return configurations;
}

void write_model(const std::string& path) {
    MTP mtp(
        MTP_Basis(MTP6_array, MTP_Basis::ORTH_OFF),
        std::make_shared<RadialBasisCinf>(8, 0.1, 3.5),
        SpeciesOrder(std::vector<Species>{1, 6, 8, 24, 25, 26, 27, 28}),
        false);
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot open model output: " + path);
    json_io::Writer writer(output);
    mtp.JsonWrite(writer);
    output << '\n';
}

std::vector<double> compute_values(
    MTP& mtp, const std::vector<Cfg>& configurations) {
    std::vector<ExtCfg> extended;
    extended.reserve(configurations.size());
    for (const auto& configuration : configurations) {
        extended.emplace_back(configuration);
        extended.back().MakeGhostAtoms(mtp.Cutoff());
        extended.back().ToRelativeSpecies(mtp.species_order());
    }

    const std::size_t radial_count =
        mtp.n_species() * mtp.n_species() *
        static_cast<std::size_t>(mtp.radial_function_count()) *
        mtp.radial_basis().size();
    const std::size_t basis_offset = radial_count + mtp.n_species();
    if (basis_offset > mtp.param_count()) {
        throw std::runtime_error("invalid MTP parameter layout");
    }
    const std::size_t feature_count = mtp.param_count() - basis_offset;
    std::size_t rows = 0;
    for (const auto& configuration : extended) rows += configuration.size();
    std::vector<double> values(rows * feature_count, 0.0);

    std::size_t row = 0;
    std::vector<double> gradients(mtp.param_count(), 0.0);
    for (const auto& configuration : extended) {
        for (const auto& nbh : configuration.nbhs()) {
            std::fill(gradients.begin(), gradients.end(), 0.0);
            mtp.AccumulateSiteEnergyGrads(nbh, gradients.data());
            std::copy(
                gradients.begin() + static_cast<std::ptrdiff_t>(basis_offset),
                gradients.end(),
                values.begin() +
                    static_cast<std::ptrdiff_t>(row * feature_count));
            ++row;
        }
    }
    return values;
}

void write_values(
    const std::string& path,
    std::size_t rows,
    std::size_t features,
    const std::vector<double>& values) {
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot open descriptor output: " + path);
    const std::uint64_t shape[2] = {
        static_cast<std::uint64_t>(rows),
        static_cast<std::uint64_t>(features)};
    output.write(reinterpret_cast<const char*>(shape), sizeof(shape));
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(double)));
}

void compute(
    const std::string& model_path,
    const std::string& input_path,
    const std::string& output_path,
    int warmups,
    int repeats) {
    if (warmups < 0 || repeats <= 0) {
        throw std::runtime_error("warmups must be non-negative and repeats positive");
    }
    std::ifstream model_input(model_path);
    if (!model_input) {
        throw std::runtime_error("cannot open model input: " + model_path);
    }
    MTP mtp;
    json_io::Reader model_reader(model_input);
    if ((model_reader >> mtp).is_error()) {
        throw std::runtime_error("cannot parse MTP model");
    }
    auto configurations = read_configurations(input_path);
    mtp.PrepareForCalc();

    std::vector<double> values;
    for (int i = 0; i < warmups; ++i) {
        values = compute_values(mtp, configurations);
    }

    std::vector<double> timings;
    timings.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        const auto start = Clock::now();
        values = compute_values(mtp, configurations);
        const auto stop = Clock::now();
        timings.push_back(std::chrono::duration<double>(stop - start).count());
    }

    std::sort(timings.begin(), timings.end());
    const double median = timings[timings.size() / 2];
    const std::size_t p95_index = static_cast<std::size_t>(
        std::ceil(0.95 * static_cast<double>(timings.size())) - 1.0);
    const double p95 = timings[std::min(p95_index, timings.size() - 1)];
    const std::size_t features =
        mtp.param_count() -
        (mtp.n_species() * mtp.n_species() *
             static_cast<std::size_t>(mtp.radial_function_count()) *
             mtp.radial_basis().size() +
         mtp.n_species());
    const std::size_t rows = values.size() / features;
    write_values(output_path, rows, features, values);

    std::cout << std::setprecision(17)
              << "{\"rows\":" << rows
              << ",\"features\":" << features
              << ",\"median_seconds\":" << median
              << ",\"p95_seconds\":" << p95
              << ",\"raw_seconds\":[";
    for (std::size_t i = 0; i < timings.size(); ++i) {
        if (i != 0) std::cout << ',';
        std::cout << timings[i];
    }
    std::cout << "]}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "generate-model") {
            write_model(argv[2]);
            return 0;
        }
        if (argc == 7 && std::string(argv[1]) == "compute") {
            compute(
                argv[2], argv[3], argv[4], std::stoi(argv[5]),
                std::stoi(argv[6]));
            return 0;
        }
        std::cerr << "usage: official_mlip4_mtp generate-model MODEL.json\n"
                  << "   or: official_mlip4_mtp compute MODEL.json INPUT.ndjson "
                     "OUTPUT.bin WARMUPS REPEATS\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "official_mlip4_mtp: " << error.what() << '\n';
        return 1;
    }
}

