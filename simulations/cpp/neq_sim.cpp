#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../../src/cpp/us.h"

namespace {

constexpr const char* kExecutablePath = "./simulations/cpp/neq_sim";

}  // namespace

struct CmdOptions {
    std::string potential = "Muller-Brown";
    bool eq_mode = false;
    bool neq_mode = false;
    bool meta_mode = false;
    bool us_mode = false;
    bool meta_fes_mode = false;
    std::string eq_out;
    std::string eq0;
    std::string eq1;
    std::string fpath;
    std::string path_out;
    std::string meta_out;
    std::string meta_hills_out;
    std::string meta_hills_in;
    std::string meta_fes_out;
    std::string out_dir;
    std::string log_path;
    std::string us_summary_out;
    std::string us_fes_out;
    std::string us_direction = "forward";

    int T_eq = 0;
    int T_neq = 0;
    int T_meta = 0;
    int T_us = 0;
    int T_us_total = 0;
    int N_neq = 0;
    int eq_nout = 0;
    int neq_nout = 0;
    int meta_nout = 0;
    int us_nout = 0;
    int eq_min_steps = 0;
    int meta_stride = 0;
    int meta_nx = 0;
    int meta_ny = 0;
    double eq_min_alpha = 0.0;
    bool eq_write_initial = false;
    bool have_eq_seed = false;
    bool have_neq_seed = false;
    unsigned int eq_seed = 123u;
    unsigned int neq_seed = 42u;

    bool have_A = false;
    bool have_B = false;
    Vec2 A{};
    Vec2 B{};

    bool have_center = false;
    Vec2 center{};
    bool have_eq_start = false;
    Vec2 eq_start{};

    bool have_meta_start = false;
    Vec2 meta_start{};

    bool have_k = false;
    double k = 0.0;
    bool have_k_midscale = false;
    double k_midscale = 1.0;

    bool have_thermal_kT = false;
    double thermal_kT = 0.0;
    bool have_dt = false;
    double dt = 0.0;
    bool have_gamma = false;
    double gamma = 0.0;

    bool have_pot_k0 = false;
    bool have_pot_x0 = false;
    bool have_pot_k1 = false;
    bool have_pot_x1 = false;
    bool have_pot_E1 = false;
    double pot_k0 = 0.0;
    double pot_x0 = 0.0;
    double pot_k1 = 0.0;
    double pot_x1 = 0.0;
    double pot_E1 = 0.0;

    bool path_only = false;
    bool check_xy = false;
    char one_dimension = 'n';

    bool have_meta_sigma_x = false;
    bool have_meta_sigma_y = false;
    bool have_meta_w0 = false;
    bool have_meta_biasfactor = false;
    bool have_meta_stride = false;
    bool have_meta_seed = false;
    double meta_sigma_x = 0.0;
    double meta_sigma_y = 0.0;
    double meta_w0 = 0.0;
    double meta_biasfactor = 0.0;
    unsigned int meta_seed = 123u;

    bool have_us_k = false;
    bool have_us_spacing = false;
    bool have_us_seed = false;
    bool have_us_grid_dx = false;
    bool have_us_grid_dy = false;
    double us_k = 0.0;
    double us_spacing = 0.0;
    double us_grid_dx = 0.0;
    double us_grid_dy = 0.0;
    unsigned int us_seed = 20260322u;

    bool have_meta_xmin = false;
    bool have_meta_xmax = false;
    bool have_meta_ymin = false;
    bool have_meta_ymax = false;
    double meta_xmin = 0.0;
    double meta_xmax = 0.0;
    double meta_ymin = 0.0;
    double meta_ymax = 0.0;
};

bool parse_vec2(const std::string& s, Vec2& out) {
    const auto comma = s.find(',');
    if (comma == std::string::npos) {
        return false;
    }
    out.x = std::stod(s.substr(0, comma));
    out.y = std::stod(s.substr(comma + 1));
    return true;
}

bool file_exists(const std::string& path) {
    return std::filesystem::exists(std::filesystem::path(path));
}

std::string now_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

std::string format_seconds(double seconds) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << seconds;
    return oss.str();
}

std::string format_vec2(Vec2 value) {
    std::ostringstream oss;
    oss << std::setprecision(10) << value.x << "," << value.y;
    return oss.str();
}

std::string format_seed_list(const std::vector<Vec2>& seeds) {
    std::ostringstream oss;
    for (size_t i = 0; i < seeds.size(); ++i) {
        if (i > 0) {
            oss << ";";
        }
        oss << format_vec2(seeds[i]);
    }
    return oss.str();
}

void write_log(const std::string& path, const std::vector<std::string>& lines) {
    if (path.empty()) {
        return;
    }
    ensure_parent_dir(path);
    std::ofstream out(path);
    for (const auto& line : lines) {
        out << line << "\n";
    }
}

void append_lines(std::vector<std::string>& dst, const std::vector<std::string>& src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

Vec2 make_one_dim_point(char one_dimension, double q) {
    if (one_dimension == 'y') {
        return {0.0, q};
    }
    return {q, 0.0};
}

Vec2 default_us_start(const SimConfig& cfg) {
    if (cfg.one_dimension == 'x' || cfg.one_dimension == 'y') {
        return make_one_dim_point(cfg.one_dimension, cfg.one_d_x0);
    }
    return cfg.A;
}

Vec2 default_us_end(const SimConfig& cfg) {
    if (cfg.one_dimension == 'x' || cfg.one_dimension == 'y') {
        return make_one_dim_point(cfg.one_dimension, cfg.one_d_x1);
    }
    return cfg.B;
}

bool reconstruct_meta_fes(const SimConfig& cfg,
                          const CmdOptions& opt,
                          const std::string& hills_in,
                          const std::string& out_path,
                          std::string& error) {
    const std::vector<MetaHillRecord> hill_records = read_meta_hills_csv(hills_in, cfg.one_dimension);
    if (hill_records.empty()) {
        error = "Failed to read metadynamics hills from " + hills_in;
        return false;
    }
    if (cfg.meta_bias_factor <= 1.0) {
        error = "Metadynamics FES reconstruction requires -meta_biasfactor > 1.";
        return false;
    }

    BiasWellTemperedMeta bias{};
    bias.one_dimension = cfg.one_dimension;
    for (const auto& hill : hill_records) {
        bias.hills.push_back(hill.gaussian);
    }

    ensure_parent_dir(out_path);
    const double wt_scale = -cfg.meta_bias_factor / (cfg.meta_bias_factor - 1.0);

    if (cfg.one_dimension == 'x' || cfg.one_dimension == 'y') {
        const double qmin = opt.have_meta_xmin ? opt.meta_xmin : std::numeric_limits<double>::quiet_NaN();
        const double qmax = opt.have_meta_xmax ? opt.meta_xmax : std::numeric_limits<double>::quiet_NaN();
        const GridSpec1D grid = auto_meta_grid_1d(
            hill_records, cfg.one_dimension, qmin, qmax, (opt.meta_nx > 0) ? opt.meta_nx : 200);

        std::vector<double> free_energy(grid.nx, std::numeric_limits<double>::infinity());
        std::vector<double> bias_values(grid.nx, 0.0);
        for (int ix = 0; ix < grid.nx; ++ix) {
            const double q = grid.xmin + grid.dx * static_cast<double>(ix);
            const Vec2 pos = make_one_dim_point(cfg.one_dimension, q);
            bias_values[ix] = bias.U(pos.x, pos.y);
            free_energy[ix] = wt_scale * bias_values[ix];
        }
        shift_free_energy_min_zero(free_energy);
        write_meta_fes_1d_csv(out_path, grid, free_energy, bias_values);
        return true;
    }

    const double xmin = opt.have_meta_xmin ? opt.meta_xmin : std::numeric_limits<double>::quiet_NaN();
    const double xmax = opt.have_meta_xmax ? opt.meta_xmax : std::numeric_limits<double>::quiet_NaN();
    const double ymin = opt.have_meta_ymin ? opt.meta_ymin : std::numeric_limits<double>::quiet_NaN();
    const double ymax = opt.have_meta_ymax ? opt.meta_ymax : std::numeric_limits<double>::quiet_NaN();
    const GridSpec2D grid = auto_meta_grid_2d(
        hill_records, xmin, xmax, ymin, ymax,
        (opt.meta_nx > 0) ? opt.meta_nx : 120,
        (opt.meta_ny > 0) ? opt.meta_ny : 120);

    const int nxy = grid.nx * grid.ny;
    std::vector<double> free_energy(nxy, std::numeric_limits<double>::infinity());
    std::vector<double> bias_values(nxy, 0.0);
    for (int iy = 0; iy < grid.ny; ++iy) {
        for (int ix = 0; ix < grid.nx; ++ix) {
            const int idx = iy * grid.nx + ix;
            const double x = grid.xmin + grid.dx * static_cast<double>(ix);
            const double y = grid.ymin + grid.dy * static_cast<double>(iy);
            bias_values[idx] = bias.U(x, y);
            free_energy[idx] = wt_scale * bias_values[idx];
        }
    }
    shift_free_energy_min_zero(free_energy);
    write_meta_fes_2d_csv(out_path, grid, free_energy, bias_values);
    return true;
}

void print_usage() {
    std::cout
        << "Usage:\n"
        << "  EQ:   " << kExecutablePath
        << " -pot \"Muller-Brown\" -center_xy x,y -eq_out <file> [-eq_start_xy x,y] [-k <value>] [-T_eq <steps>] [-eq_nout <n>] [-eq_minimize <steps>] [-eq_min_alpha <alpha>] [-eq_write_initial] [-eq_seed <seed>] -out_dir <dir>\n"
        << "  NEQ:  " << kExecutablePath
        << " -pot \"Muller-Brown\" -eq0 <file> -eq1 <file> -N_neq <ntraj> -T_neq <steps> [-neq_nout <n>] [-neq_seed <seed>] [-fpath <file>] [-path_out <file>] [-A_center x,y -B_center x,y] [-k <value>] [-k_midscale <value>] [-one-dimension x|y] -out_dir <dir>\n"
        << "  META: " << kExecutablePath
        << " -pot \"Muller-Brown\" -meta_start_xy x,y -T_meta <steps> [-meta_out <file>] [-meta_hills_out <file>] [-meta_fes_out <file>] [-meta_sigma_x <v>] [-meta_sigma_y <v>] [-meta_w0 <v>] [-meta_biasfactor <v>] [-meta_stride <steps>] [-meta_nout <n>] [-meta_seed <seed>] [-one-dimension x|y] -out_dir <dir>\n"
        << "  US:   " << kExecutablePath
        << " -pot \"Muller-Brown\" -us_mode (-T_us <steps_per_window> | -T_us_total <total_steps>) -us_k <stiffness> -us_spacing <dx> [-us_nout <n>] [-us_direction forward|backward|bidirectional] [-us_summary_out <file>] [-us_fes_out <file>] [-us_grid_dx <dx>] [-us_grid_dy <dy>] [-us_seed <seed>] [-A_center x,y -B_center x,y] [-one-dimension x|y] -out_dir <dir>\n"
        << "  META_FES: " << kExecutablePath
        << " -pot \"Muller-Brown\" -meta_fes_mode -meta_hills_in <file> -meta_biasfactor <value> [-meta_fes_out <file>] [-meta_xmin <v>] [-meta_xmax <v>] [-meta_ymin <v>] [-meta_ymax <v>] [-meta_nx <n>] [-meta_ny <n>] [-one-dimension x|y] -out_dir <dir>\n"
        << "  PATH: " << kExecutablePath
        << " -pot \"Muller-Brown\" -path_out <file> [-A_center x,y -B_center x,y] [-one-dimension x|y] [-k <value>] [-k_midscale <value>] -out_dir <dir>\n"
        << "  CHECK: " << kExecutablePath << " -pot \"Muller-Brown\" -check_xy\n"
        << "  Thermal parameters: [-thermal_kT <value>] [-dt <value>] [-gamma <value>] set dynamics temperature, timestep, and friction\n"
        << "  1D potential params for -pot \"Double-well_1D\": [-k0 v] [-x0 v] [-k1 v] [-x1 v] [-E1 v] with -one-dimension x|y\n"
        << "  LOG: add -log <file> to write a summary of the run\n";
}

bool parse_args(int argc, char** argv, CmdOptions& opt) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](std::string& dst) -> bool {
            if (i + 1 >= argc) {
                return false;
            }
            dst = argv[++i];
            return true;
        };
        auto next_int = [&](int& dst) -> bool {
            if (i + 1 >= argc) {
                return false;
            }
            dst = std::stoi(argv[++i]);
            return true;
        };
        auto next_double = [&](double& dst) -> bool {
            if (i + 1 >= argc) {
                return false;
            }
            dst = std::stod(argv[++i]);
            return true;
        };
        auto next_uint = [&](unsigned int& dst) -> bool {
            if (i + 1 >= argc) {
                return false;
            }
            dst = static_cast<unsigned int>(std::stoul(argv[++i]));
            return true;
        };

        if (arg == "-pot") {
            if (!next(opt.potential)) return false;
        } else if (arg == "-center_xy") {
            std::string val;
            if (!next(val)) return false;
            if (!parse_vec2(val, opt.center)) return false;
            opt.have_center = true;
            opt.eq_mode = true;
        } else if (arg == "-eq_start_xy") {
            std::string val;
            if (!next(val)) return false;
            if (!parse_vec2(val, opt.eq_start)) return false;
            opt.have_eq_start = true;
            opt.eq_mode = true;
        } else if (arg == "-eq_out") {
            if (!next(opt.eq_out)) return false;
            opt.eq_mode = true;
        } else if (arg == "-meta_start_xy") {
            std::string val;
            if (!next(val)) return false;
            if (!parse_vec2(val, opt.meta_start)) return false;
            opt.have_meta_start = true;
            opt.meta_mode = true;
        } else if (arg == "-meta_out") {
            if (!next(opt.meta_out)) return false;
            opt.meta_mode = true;
        } else if (arg == "-meta_hills_out") {
            if (!next(opt.meta_hills_out)) return false;
            opt.meta_mode = true;
        } else if (arg == "-meta_hills_in") {
            if (!next(opt.meta_hills_in)) return false;
            opt.meta_fes_mode = true;
        } else if (arg == "-meta_fes_out") {
            if (!next(opt.meta_fes_out)) return false;
        } else if (arg == "-meta_fes_mode") {
            opt.meta_fes_mode = true;
        } else if (arg == "-meta_xmin") {
            if (!next_double(opt.meta_xmin)) return false;
            opt.have_meta_xmin = true;
        } else if (arg == "-meta_xmax") {
            if (!next_double(opt.meta_xmax)) return false;
            opt.have_meta_xmax = true;
        } else if (arg == "-meta_ymin") {
            if (!next_double(opt.meta_ymin)) return false;
            opt.have_meta_ymin = true;
        } else if (arg == "-meta_ymax") {
            if (!next_double(opt.meta_ymax)) return false;
            opt.have_meta_ymax = true;
        } else if (arg == "-meta_nx") {
            if (!next_int(opt.meta_nx)) return false;
        } else if (arg == "-meta_ny") {
            if (!next_int(opt.meta_ny)) return false;
        } else if (arg == "-T_meta") {
            if (!next_int(opt.T_meta)) return false;
            opt.meta_mode = true;
        } else if (arg == "-meta_nout") {
            if (!next_int(opt.meta_nout)) return false;
            opt.meta_mode = true;
        } else if (arg == "-meta_sigma_x") {
            if (!next_double(opt.meta_sigma_x)) return false;
            opt.have_meta_sigma_x = true;
            opt.meta_mode = true;
        } else if (arg == "-meta_sigma_y") {
            if (!next_double(opt.meta_sigma_y)) return false;
            opt.have_meta_sigma_y = true;
            opt.meta_mode = true;
        } else if (arg == "-meta_w0") {
            if (!next_double(opt.meta_w0)) return false;
            opt.have_meta_w0 = true;
            opt.meta_mode = true;
        } else if (arg == "-meta_biasfactor") {
            if (!next_double(opt.meta_biasfactor)) return false;
            opt.have_meta_biasfactor = true;
        } else if (arg == "-meta_stride") {
            if (!next_int(opt.meta_stride)) return false;
            opt.have_meta_stride = true;
            opt.meta_mode = true;
        } else if (arg == "-meta_seed") {
            if (!next_uint(opt.meta_seed)) return false;
            opt.have_meta_seed = true;
            opt.meta_mode = true;
        } else if (arg == "-eq0") {
            if (!next(opt.eq0)) return false;
            opt.neq_mode = true;
        } else if (arg == "-eq1") {
            if (!next(opt.eq1)) return false;
            opt.neq_mode = true;
        } else if (arg == "-N_neq") {
            if (!next_int(opt.N_neq)) return false;
            opt.neq_mode = true;
        } else if (arg == "-T_eq") {
            if (!next_int(opt.T_eq)) return false;
            opt.eq_mode = true;
        } else if (arg == "-T_us" || arg == "-T_aus") {
            if (!next_int(opt.T_us)) return false;
            opt.us_mode = true;
        } else if (arg == "-T_us_total" || arg == "-T_aus_total") {
            if (!next_int(opt.T_us_total)) return false;
            opt.us_mode = true;
        } else if (arg == "-eq_nout") {
            if (!next_int(opt.eq_nout)) return false;
            opt.eq_mode = true;
        } else if (arg == "-us_nout" || arg == "-aus_nout") {
            if (!next_int(opt.us_nout)) return false;
            opt.us_mode = true;
        } else if (arg == "-eq_minimize") {
            if (!next_int(opt.eq_min_steps)) return false;
            opt.eq_mode = true;
        } else if (arg == "-eq_min_alpha") {
            if (!next_double(opt.eq_min_alpha)) return false;
            opt.eq_mode = true;
        } else if (arg == "-eq_write_initial") {
            opt.eq_write_initial = true;
            opt.eq_mode = true;
        } else if (arg == "-eq_seed") {
            if (!next_uint(opt.eq_seed)) return false;
            opt.have_eq_seed = true;
            opt.eq_mode = true;
        } else if (arg == "-neq_nout") {
            if (!next_int(opt.neq_nout)) return false;
            opt.neq_mode = true;
        } else if (arg == "-T_neq") {
            if (!next_int(opt.T_neq)) return false;
            opt.neq_mode = true;
        } else if (arg == "-neq_seed") {
            if (!next_uint(opt.neq_seed)) return false;
            opt.have_neq_seed = true;
            opt.neq_mode = true;
        } else if (arg == "-fpath") {
            if (!next(opt.fpath)) return false;
            opt.neq_mode = true;
        } else if (arg == "-path_out") {
            if (!next(opt.path_out)) return false;
        } else if (arg == "-path_only") {
            opt.path_only = true;
        } else if (arg == "-us_mode" || arg == "-aus_mode") {
            opt.us_mode = true;
        } else if (arg == "-us_direction") {
            if (!next(opt.us_direction)) return false;
            opt.us_mode = true;
        } else if (arg == "-us_k" || arg == "-aus_k") {
            if (!next_double(opt.us_k)) return false;
            opt.have_us_k = true;
            opt.us_mode = true;
        } else if (arg == "-us_spacing" || arg == "-aus_spacing") {
            if (!next_double(opt.us_spacing)) return false;
            opt.have_us_spacing = true;
            opt.us_mode = true;
        } else if (arg == "-us_summary_out" || arg == "-aus_summary_out") {
            if (!next(opt.us_summary_out)) return false;
            opt.us_mode = true;
        } else if (arg == "-us_fes_out" || arg == "-aus_fes_out") {
            if (!next(opt.us_fes_out)) return false;
            opt.us_mode = true;
        } else if (arg == "-us_grid_dx" || arg == "-aus_grid_dx") {
            if (!next_double(opt.us_grid_dx)) return false;
            opt.have_us_grid_dx = true;
            opt.us_mode = true;
        } else if (arg == "-us_grid_dy" || arg == "-aus_grid_dy") {
            if (!next_double(opt.us_grid_dy)) return false;
            opt.have_us_grid_dy = true;
            opt.us_mode = true;
        } else if (arg == "-us_seed" || arg == "-aus_seed") {
            if (!next_uint(opt.us_seed)) return false;
            opt.have_us_seed = true;
            opt.us_mode = true;
        } else if (arg == "-us_out_dir" || arg == "-aus_out_dir") {
            if (!next(opt.out_dir)) return false;
            opt.us_mode = true;
        } else if (arg == "-check_xy") {
            opt.check_xy = true;
        } else if (arg == "-one-dimension") {
            std::string val;
            if (!next(val)) return false;
            if (val != "x" && val != "y") {
                return false;
            }
            opt.one_dimension = val[0];
        } else if (arg == "-log") {
            if (!next(opt.log_path)) return false;
        } else if (arg == "-out_dir") {
            if (!next(opt.out_dir)) return false;
        } else if (arg == "-A_center") {
            std::string val;
            if (!next(val)) return false;
            if (!parse_vec2(val, opt.A)) return false;
            opt.have_A = true;
        } else if (arg == "-B_center") {
            std::string val;
            if (!next(val)) return false;
            if (!parse_vec2(val, opt.B)) return false;
            opt.have_B = true;
        } else if (arg == "-k") {
            if (!next_double(opt.k)) return false;
            opt.have_k = true;
        } else if (arg == "-k_midscale") {
            if (!next_double(opt.k_midscale)) return false;
            opt.have_k_midscale = true;
        } else if (arg == "-thermal_kT") {
            if (!next_double(opt.thermal_kT)) return false;
            opt.have_thermal_kT = true;
        } else if (arg == "-dt") {
            if (!next_double(opt.dt)) return false;
            opt.have_dt = true;
        } else if (arg == "-gamma") {
            if (!next_double(opt.gamma)) return false;
            opt.have_gamma = true;
        } else if (arg == "-k0") {
            if (!next_double(opt.pot_k0)) return false;
            opt.have_pot_k0 = true;
        } else if (arg == "-x0") {
            if (!next_double(opt.pot_x0)) return false;
            opt.have_pot_x0 = true;
        } else if (arg == "-k1") {
            if (!next_double(opt.pot_k1)) return false;
            opt.have_pot_k1 = true;
        } else if (arg == "-x1") {
            if (!next_double(opt.pot_x1)) return false;
            opt.have_pot_x1 = true;
        } else if (arg == "-E1") {
            if (!next_double(opt.pot_E1)) return false;
            opt.have_pot_E1 = true;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            return false;
        }
    }

    return true;
}

int main(int argc, char** argv) {
    const auto t_start = std::chrono::steady_clock::now();
    CmdOptions opt{};
    if (!parse_args(argc, argv, opt)) {
        print_usage();
        return 1;
    }

    if (!opt.eq_mode && !opt.neq_mode && !opt.meta_mode && !opt.us_mode &&
        !opt.meta_fes_mode && !opt.path_only && !opt.check_xy) {
        print_usage();
        return 1;
    }

    if (!opt.check_xy && opt.out_dir.empty()) {
        std::cerr << "All modes require -out_dir.\n";
        return 1;
    }

    SimConfig cfg{};
    if (opt.have_A) cfg.A = opt.A;
    if (opt.have_B) cfg.B = opt.B;
    cfg.out_dir = opt.out_dir;

    if (opt.potential == "Muller-Brown") {
        cfg.potential = SimConfig::PotentialType::MullerBrown;
    } else if (opt.potential == "Six-hump_camel") {
        cfg.potential = SimConfig::PotentialType::SixHumpCamel;
    } else if (opt.potential == "Three_wells") {
        cfg.potential = SimConfig::PotentialType::ThreeWell;
    } else if (opt.potential == "Double-well_1D") {
        cfg.potential = SimConfig::PotentialType::DoubleWell1D;
    } else {
        std::cerr << "Unknown potential: " << opt.potential << "\n";
        return 1;
    }

    if (opt.have_pot_k0) cfg.one_d_k0 = opt.pot_k0;
    if (opt.have_pot_x0) cfg.one_d_x0 = opt.pot_x0;
    if (opt.have_pot_k1) cfg.one_d_k1 = opt.pot_k1;
    if (opt.have_pot_x1) cfg.one_d_x1 = opt.pot_x1;
    if (opt.have_pot_E1) cfg.one_d_E1 = opt.pot_E1;
    if (opt.have_thermal_kT) cfg.kT = opt.thermal_kT;
    if (opt.have_dt) cfg.dt = opt.dt;
    if (opt.have_gamma) cfg.gamma = opt.gamma;
    if (opt.have_k) cfg.k = opt.k;
    if (opt.have_k_midscale) cfg.k_midscale = opt.k_midscale;
    if (opt.one_dimension != 'n') cfg.one_dimension = opt.one_dimension;

    if (opt.T_neq > 0) cfg.n_neq_steps = opt.T_neq;
    if (opt.T_meta > 0) cfg.n_meta_steps = opt.T_meta;
    if (opt.eq_nout > 0) cfg.eq_output_samples = opt.eq_nout;
    if (opt.neq_nout > 0) {
        cfg.neq_output_stride = std::max(1, cfg.n_neq_steps / opt.neq_nout);
    }
    if (opt.have_meta_sigma_x) cfg.meta_sigma_x = opt.meta_sigma_x;
    if (opt.have_meta_sigma_y) cfg.meta_sigma_y = opt.meta_sigma_y;
    if (opt.have_meta_sigma_x && !opt.have_meta_sigma_y) cfg.meta_sigma_y = opt.meta_sigma_x;
    if (opt.have_meta_sigma_y && !opt.have_meta_sigma_x) cfg.meta_sigma_x = opt.meta_sigma_y;
    if (opt.have_meta_w0) cfg.meta_initial_height = opt.meta_w0;
    if (opt.have_meta_biasfactor) cfg.meta_bias_factor = opt.meta_biasfactor;
    if (opt.have_meta_stride) cfg.meta_deposition_stride = opt.meta_stride;
    if (opt.have_meta_seed) cfg.meta_seed = opt.meta_seed;
    if (opt.have_eq_seed) cfg.eq_seed = opt.eq_seed;
    if (opt.have_neq_seed) cfg.neq_seed = opt.neq_seed;
    if (opt.meta_nout > 0) {
        cfg.meta_output_stride = std::max(1, cfg.n_meta_steps / opt.meta_nout);
    }
    if (opt.eq_min_steps > 0) cfg.eq_minimize_steps = opt.eq_min_steps;
    if (opt.eq_min_alpha > 0.0) cfg.eq_minimize_alpha = opt.eq_min_alpha;
    if (opt.eq_write_initial) cfg.eq_write_initial = true;

    if (cfg.kT <= 0.0) {
        std::cerr << "-thermal_kT must be > 0.\n";
        return 1;
    }
    if (cfg.dt <= 0.0) {
        std::cerr << "-dt must be > 0.\n";
        return 1;
    }
    if (cfg.gamma <= 0.0) {
        std::cerr << "-gamma must be > 0.\n";
        return 1;
    }
    if (cfg.k <= 0.0) {
        std::cerr << "-k must be > 0.\n";
        return 1;
    }

    if (opt.check_xy) {
        if (cfg.potential == SimConfig::PotentialType::DoubleWell1D) {
            if (cfg.one_dimension == 'n') {
                cfg.one_dimension = 'x';
            }
            std::cout << "1D U(-1.0,0.0)=" << potential_U(cfg, -1.0, 0.0) << "\n";
            std::cout << "1D U(0.0,0.0)=" << potential_U(cfg, 0.0, 0.0) << "\n";
            std::cout << "1D U(1.0,0.0)=" << potential_U(cfg, 1.0, 0.0) << "\n";
        } else {
            const Vec2 p1{-0.6, 1.4};
            const Vec2 p2{0.0, 0.5};
            std::cout << "MB U(-0.6,1.4)=" << MullerBrown::U(p1.x, p1.y) << "\n";
            std::cout << "MB U(1.4,-0.6)=" << MullerBrown::U(p1.y, p1.x) << "\n";
            std::cout << "MB U(0.0,0.5)=" << MullerBrown::U(p2.x, p2.y) << "\n";
            std::cout << "MB U(0.5,0.0)=" << MullerBrown::U(p2.y, p2.x) << "\n";
        }
        return 0;
    }

    if (cfg.potential == SimConfig::PotentialType::DoubleWell1D && cfg.one_dimension == 'n') {
        std::cerr << "Double-well_1D requires -one-dimension x or -one-dimension y.\n";
        return 1;
    }

    auto resolve_log_path = [&](const std::string& path) -> std::string {
        if (path.empty()) {
            return (std::filesystem::path(cfg.out_dir) / "run.log").string();
        }
        return std::filesystem::path(path).string();
    };
    auto resolve_out_path = [&](const std::string& path, const std::string& fallback) -> std::filesystem::path {
        if (path.empty()) {
            return std::filesystem::path(cfg.out_dir) / fallback;
        }
        const std::filesystem::path p(path);
        if (p.is_absolute()) {
            return p;
        }
        return std::filesystem::path(cfg.out_dir) / p;
    };

    if (opt.meta_fes_mode) {
        if (opt.meta_hills_in.empty()) {
            std::cerr << "META_FES mode requires -meta_hills_in.\n";
            return 1;
        }
        if (!file_exists(opt.meta_hills_in)) {
            std::cerr << "Metadynamics hills file not found: " << opt.meta_hills_in << "\n";
            return 1;
        }
        if (cfg.meta_bias_factor <= 1.0) {
            std::cerr << "META_FES mode requires -meta_biasfactor > 1.\n";
            return 1;
        }

        std::filesystem::create_directories(cfg.out_dir);
        const std::filesystem::path meta_fes_out = resolve_out_path(opt.meta_fes_out, "meta_fes.csv");
        std::string error;
        if (!reconstruct_meta_fes(cfg, opt, opt.meta_hills_in, meta_fes_out.string(), error)) {
            std::cerr << error << "\n";
            return 1;
        }

        const auto t_end = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        std::vector<std::string> log_lines{
            "mode=META_FES",
            "potential=" + opt.potential,
            "meta_hills_in=" + opt.meta_hills_in,
            "meta_fes_out=" + meta_fes_out.string(),
            "meta_biasfactor=" + std::to_string(cfg.meta_bias_factor),
            "one_dimension=" + std::string(1, cfg.one_dimension),
            "started_at=" + now_timestamp(),
            "elapsed_sec=" + format_seconds(elapsed)
        };
        BenchmarkSummary benchmark{};
        benchmark.method = "WT_META_ANALYZE";
        benchmark.n_hills = static_cast<int>(read_meta_hills_csv(opt.meta_hills_in, cfg.one_dimension).size());
        benchmark.hyperparameters = "biasfactor=" + std::to_string(cfg.meta_bias_factor);
        benchmark.fes_path = meta_fes_out.string();
        append_lines(log_lines, benchmark_summary_lines(benchmark));
        write_log(resolve_log_path(opt.log_path), log_lines);
        std::cout << "Wrote metadynamics FES to " << meta_fes_out.string() << "\n";
        return 0;
    }

    if (opt.meta_mode) {
        if (cfg.n_meta_steps <= 0) {
            std::cerr << "META mode requires -T_meta > 0.\n";
            return 1;
        }
        if (cfg.meta_sigma_x <= 0.0 || cfg.meta_sigma_y <= 0.0) {
            std::cerr << "META mode requires positive -meta_sigma_x and -meta_sigma_y.\n";
            return 1;
        }
        if (cfg.meta_initial_height <= 0.0) {
            std::cerr << "META mode requires -meta_w0 > 0.\n";
            return 1;
        }
        if (cfg.meta_bias_factor <= 1.0) {
            std::cerr << "META mode requires -meta_biasfactor > 1.\n";
            return 1;
        }
        if (cfg.meta_deposition_stride <= 0) {
            std::cerr << "META mode requires -meta_stride > 0.\n";
            return 1;
        }

        std::filesystem::create_directories(cfg.out_dir);

        const Vec2 meta_start = opt.have_meta_start ? opt.meta_start : cfg.A;
        const std::filesystem::path traj_out = resolve_out_path(opt.meta_out, "meta_traj.csv");
        const std::filesystem::path hills_out = resolve_out_path(opt.meta_hills_out, "meta_hills.csv");

        MetaRunSummary summary = run_well_tempered_meta_write(
            cfg, traj_out.string(), hills_out.string(), meta_start, cfg.meta_seed);

        std::string meta_fes_path = "none";
        if (!opt.meta_fes_out.empty()) {
            const std::filesystem::path meta_fes_out = resolve_out_path(opt.meta_fes_out, "meta_fes.csv");
            std::string error;
            if (!reconstruct_meta_fes(cfg, opt, hills_out.string(), meta_fes_out.string(), error)) {
                std::cerr << error << "\n";
                return 1;
            }
            meta_fes_path = meta_fes_out.string();
        }

        const auto t_end = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        std::vector<std::string> log_lines{
            "mode=WT_META",
            "potential=" + opt.potential,
            "meta_start=" + format_vec2(meta_start),
            "meta_out=" + traj_out.string(),
            "meta_hills_out=" + hills_out.string(),
            "meta_fes_out=" + meta_fes_path,
            "T_meta=" + std::to_string(cfg.n_meta_steps),
            "meta_w0=" + std::to_string(cfg.meta_initial_height),
            "meta_sigma_x=" + std::to_string(cfg.meta_sigma_x),
            "meta_sigma_y=" + std::to_string(cfg.meta_sigma_y),
            "meta_biasfactor=" + std::to_string(cfg.meta_bias_factor),
            "meta_stride=" + std::to_string(cfg.meta_deposition_stride),
            "meta_output_stride=" + std::to_string(cfg.meta_output_stride),
            "meta_seed=" + std::to_string(cfg.meta_seed),
            "thermal_kT=" + std::to_string(cfg.kT),
            "one_dimension=" + std::string(1, cfg.one_dimension),
            "final_x=" + std::to_string(summary.final_pos.x),
            "final_y=" + std::to_string(summary.final_pos.y),
            "final_base_u=" + std::to_string(summary.final_base_u),
            "final_meta_u=" + std::to_string(summary.final_meta_u),
            "n_hills=" + std::to_string(summary.hills_deposited),
            "started_at=" + now_timestamp(),
            "elapsed_sec=" + format_seconds(elapsed)
        };
        BenchmarkSummary benchmark{};
        benchmark.method = "WT_META";
        benchmark.n_steps_total = cfg.n_meta_steps;
        benchmark.n_force_evals_est = estimate_force_evals(cfg.n_meta_steps);
        benchmark.n_hills = summary.hills_deposited;
        benchmark.seed = cfg.meta_seed;
        benchmark.hyperparameters =
            "w0=" + std::to_string(cfg.meta_initial_height) +
            ";sigma_x=" + std::to_string(cfg.meta_sigma_x) +
            ";sigma_y=" + std::to_string(cfg.meta_sigma_y) +
            ";biasfactor=" + std::to_string(cfg.meta_bias_factor) +
            ";stride=" + std::to_string(cfg.meta_deposition_stride);
        benchmark.fes_path = meta_fes_path;
        append_lines(log_lines, benchmark_summary_lines(benchmark));
        write_log(resolve_log_path(opt.log_path), log_lines);
        std::cout << "Wrote metadynamics outputs to " << cfg.out_dir << "\n";
        return 0;
    }

    if (opt.us_mode) {
        const double us_k = opt.have_us_k ? opt.us_k : cfg.k;
        const double spacing = opt.have_us_spacing ? opt.us_spacing : 0.2;
        const int steps_per_window = (opt.T_us > 0) ? opt.T_us : cfg.n_eq_steps;
        const int total_steps = opt.T_us_total;
        const int output_samples = (opt.us_nout > 0) ? opt.us_nout : cfg.eq_output_samples;

        if (us_k <= 0.0) {
            std::cerr << "US mode requires -us_k > 0.\n";
            return 1;
        }
        if (spacing <= 0.0) {
            std::cerr << "US mode requires -us_spacing > 0.\n";
            return 1;
        }
        if (total_steps <= 0 && steps_per_window <= 0) {
            std::cerr << "US mode requires -T_us > 0 or -T_us_total > 0.\n";
            return 1;
        }
        if (total_steps < 0) {
            std::cerr << "US mode requires -T_us_total >= 0.\n";
            return 1;
        }
        if (opt.us_direction != "forward" && opt.us_direction != "backward" &&
            opt.us_direction != "bidirectional") {
            std::cerr << "US mode requires -us_direction forward, backward, or bidirectional.\n";
            return 1;
        }

        UmbrellaSamplingConfig us_cfg{};
        us_cfg.steps_per_window = steps_per_window;
        us_cfg.total_steps = total_steps;
        us_cfg.output_samples = std::max(1, output_samples);
        us_cfg.k = us_k;
        us_cfg.spacing = spacing;
        us_cfg.grid_dx = opt.have_us_grid_dx ? opt.us_grid_dx : 0.0;
        us_cfg.grid_dy = opt.have_us_grid_dy ? opt.us_grid_dy : 0.0;
        us_cfg.base_seed = opt.have_us_seed ? opt.us_seed : 20260322u;
        us_cfg.direction = opt.us_direction;
        us_cfg.start = default_us_start(cfg);
        us_cfg.end = default_us_end(cfg);
        us_cfg.out_dir = cfg.out_dir;
        us_cfg.summary_out = resolve_out_path(opt.us_summary_out, "us_windows.csv").string();
        us_cfg.fes_out = resolve_out_path(opt.us_fes_out, "us_fes.csv").string();

        std::filesystem::create_directories(cfg.out_dir);
        UmbrellaSamplingSummary summary = run_umbrella_sampling(cfg, us_cfg);

        const auto t_end = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        std::vector<std::string> log_lines{
            "mode=US",
            "potential=" + opt.potential,
            "thermal_kT=" + std::to_string(cfg.kT),
            "one_dimension=" + std::string(1, cfg.one_dimension),
            "T_us=" + std::to_string(us_cfg.steps_per_window),
            "T_us_total=" + std::to_string(us_cfg.total_steps),
            "us_nout=" + std::to_string(us_cfg.output_samples),
            "us_k=" + std::to_string(us_cfg.k),
            "us_spacing=" + std::to_string(us_cfg.spacing),
            "us_direction=" + us_cfg.direction,
            "us_seed=" + std::to_string(us_cfg.base_seed),
            "us_start=" + format_vec2(us_cfg.start),
            "us_end=" + format_vec2(us_cfg.end),
            "us_summary_out=" + summary.summary_path,
            "us_fes_out=" + summary.fes_path,
            "started_at=" + now_timestamp(),
            "elapsed_sec=" + format_seconds(elapsed)
        };
        BenchmarkSummary benchmark{};
        benchmark.method = "US";
        benchmark.n_steps_total = summary.n_steps_total;
        benchmark.n_force_evals_est = summary.n_force_evals_est;
        benchmark.n_windows = summary.n_windows_run;
        benchmark.seed = us_cfg.base_seed;
        benchmark.hyperparameters =
            "k=" + std::to_string(us_cfg.k) +
            ";spacing=" + std::to_string(us_cfg.spacing) +
            ";steps_per_window=" + std::to_string(us_cfg.steps_per_window) +
            ";total_steps=" + std::to_string(us_cfg.total_steps) +
            ";direction=" + us_cfg.direction;
        benchmark.fes_path = summary.fes_path;
        append_lines(log_lines, benchmark_summary_lines(benchmark));
        write_log(resolve_log_path(opt.log_path), log_lines);
        std::cout << "Wrote umbrella sampling outputs to " << cfg.out_dir << "\n";
        return 0;
    }

    if (opt.path_only) {
        if (opt.path_out.empty()) {
            std::cerr << "PATH mode requires -path_out.\n";
            return 1;
        }
        if (!opt.fpath.empty()) {
            if (!file_exists(opt.fpath)) {
                std::cerr << "Path file not found: " << opt.fpath << "\n";
                return 1;
            }
            cfg.path_csv = opt.fpath;
        }

        PathData path = build_path(cfg);
        const std::filesystem::path path_out = resolve_out_path(opt.path_out, "neq_path.csv");
        ensure_parent_dir(path_out.string());
        write_path_csv(path_out.string(), path);

        const auto t_end = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        const std::string log_path = resolve_log_path(opt.log_path);
        std::vector<std::string> log_lines{
            "mode=PATH",
            "potential=" + opt.potential,
            "pot_k0=" + std::to_string(cfg.one_d_k0),
            "pot_x0=" + std::to_string(cfg.one_d_x0),
            "pot_k1=" + std::to_string(cfg.one_d_k1),
            "pot_x1=" + std::to_string(cfg.one_d_x1),
            "pot_E1=" + std::to_string(cfg.one_d_E1),
            "thermal_kT=" + std::to_string(cfg.kT),
            "k=" + std::to_string(cfg.k),
            "k_midscale=" + std::to_string(cfg.k_midscale),
            "path_in=" + (opt.fpath.empty() ? std::string("none") : opt.fpath),
            "path_out=" + path_out.string(),
            "A_center=" + format_vec2(cfg.A),
            "B_center=" + format_vec2(cfg.B),
            "n_path_points=" + std::to_string(static_cast<int>(path.points.size())),
            "started_at=" + now_timestamp(),
            "elapsed_sec=" + format_seconds(elapsed)
        };
        write_log(log_path, log_lines);
        std::cout << "Wrote path to " << path_out.string() << "\n";
        return 0;
    }

    if (opt.eq_mode) {
        if (opt.eq_out.empty() || !opt.have_center) {
            std::cerr << "EQ mode requires -center_xy and -eq_out.\n";
            return 1;
        }
        if (opt.T_eq > 0) {
            cfg.n_eq_steps = opt.T_eq;
        }

        const std::filesystem::path eq_out = resolve_out_path(opt.eq_out, "eq.csv");
        run_eq_harmonic_write(
            cfg,
            eq_out.string(),
            opt.center,
            cfg.k,
            cfg.eq_seed,
            opt.have_eq_start,
            opt.eq_start
        );

        const auto t_end = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        std::vector<std::string> log_lines{
            "mode=EQ",
            "potential=" + opt.potential,
            "pot_k0=" + std::to_string(cfg.one_d_k0),
            "pot_x0=" + std::to_string(cfg.one_d_x0),
            "pot_k1=" + std::to_string(cfg.one_d_k1),
            "pot_x1=" + std::to_string(cfg.one_d_x1),
            "pot_E1=" + std::to_string(cfg.one_d_E1),
            "thermal_kT=" + std::to_string(cfg.kT),
            "k=" + std::to_string(cfg.k),
            "x0=" + std::to_string(opt.center.x),
            "y0=" + std::to_string(opt.center.y),
            "eq_start_x=" + std::to_string(opt.have_eq_start ? opt.eq_start.x : opt.center.x),
            "eq_start_y=" + std::to_string(opt.have_eq_start ? opt.eq_start.y : opt.center.y),
            "eq_out=" + eq_out.string(),
            "one_dimension=" + std::string(1, cfg.one_dimension),
            "eq_seed=" + std::to_string(cfg.eq_seed),
            "eq_minimize_steps=" + std::to_string(cfg.eq_minimize_steps),
            "eq_minimize_alpha=" + std::to_string(cfg.eq_minimize_alpha),
            "T_eq=" + std::to_string(cfg.n_eq_steps),
            "started_at=" + now_timestamp(),
            "elapsed_sec=" + format_seconds(elapsed)
        };
        BenchmarkSummary benchmark{};
        benchmark.method = "EQ";
        benchmark.n_steps_total = cfg.n_eq_steps;
        benchmark.n_force_evals_est = estimate_force_evals(cfg.n_eq_steps);
        benchmark.hyperparameters = "k=" + std::to_string(cfg.k);
        append_lines(log_lines, benchmark_summary_lines(benchmark));
        write_log(resolve_log_path(opt.log_path), log_lines);
        std::cout << "Wrote EQ samples to " << eq_out.string() << "\n";
        return 0;
    }

    if (opt.neq_mode) {
        if (opt.eq0.empty() || opt.eq1.empty() || opt.N_neq <= 0 || opt.T_neq <= 0) {
            std::cerr << "NEQ mode requires -eq0, -eq1, -N_neq, and -T_neq.\n";
            return 1;
        }
        const std::filesystem::path eq0 = std::filesystem::path(opt.eq0);
        const std::filesystem::path eq1 = std::filesystem::path(opt.eq1);
        if (!file_exists(eq0.string()) || !file_exists(eq1.string())) {
            std::cerr << "EQ files not found: " << eq0.string() << " or " << eq1.string() << "\n";
            return 1;
        }

        cfg.n_neq_traj = opt.N_neq;
        cfg.n_neq_steps = opt.T_neq;
        if (!opt.fpath.empty()) {
            const std::filesystem::path fpath = std::filesystem::path(opt.fpath);
            if (!file_exists(fpath.string())) {
                std::cerr << "Path file not found: " << fpath.string() << "\n";
                return 1;
            }
            cfg.path_csv = fpath.string();
        }

        std::filesystem::create_directories(cfg.out_dir);

        PathData path = build_path(cfg);
        PathData path_out_data = downsample_path(path, cfg.neq_output_stride);
        const std::filesystem::path path_out = opt.path_out.empty()
            ? (std::filesystem::path(cfg.out_dir) / "neq_path.csv")
            : resolve_out_path(opt.path_out, "neq_path.csv");
        ensure_parent_dir(path_out.string());
        write_path_csv(path_out.string(), path_out_data);

        std::vector<Vec2> eq0_samples = read_eq_samples(eq0.string());
        std::vector<Vec2> eq1_samples = read_eq_samples(eq1.string());
        if (eq0_samples.empty() || eq1_samples.empty()) {
            std::cerr << "Failed to read EQ samples from files.\n";
            return 1;
        }

        for (int i = 0; i < cfg.n_neq_traj; ++i) {
            run_neq_forward(cfg, i, eq0_samples, path);
            run_neq_backward(cfg, i, eq1_samples, path);
        }

        const auto t_end = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        std::vector<std::string> log_lines{
            "mode=NEQ",
            "potential=" + opt.potential,
            "pot_k0=" + std::to_string(cfg.one_d_k0),
            "pot_x0=" + std::to_string(cfg.one_d_x0),
            "pot_k1=" + std::to_string(cfg.one_d_k1),
            "pot_x1=" + std::to_string(cfg.one_d_x1),
            "pot_E1=" + std::to_string(cfg.one_d_E1),
            "thermal_kT=" + std::to_string(cfg.kT),
            "k=" + std::to_string(cfg.k),
            "k_midscale=" + std::to_string(cfg.k_midscale),
            "A_center=" + format_vec2(cfg.A),
            "B_center=" + format_vec2(cfg.B),
            "eq0=" + eq0.string(),
            "eq1=" + eq1.string(),
            "path_in=" + (cfg.path_csv.empty() ? std::string("none") : cfg.path_csv),
            "path_out=" + path_out.string(),
            "N_neq=" + std::to_string(cfg.n_neq_traj),
            "T_neq=" + std::to_string(cfg.n_neq_steps),
            "neq_output_stride=" + std::to_string(cfg.neq_output_stride),
            "neq_seed=" + std::to_string(cfg.neq_seed),
            "one_dimension=" + std::string(1, cfg.one_dimension),
            "started_at=" + now_timestamp(),
            "elapsed_sec=" + format_seconds(elapsed)
        };
        BenchmarkSummary benchmark{};
        benchmark.method = "NEQ";
        benchmark.n_steps_total = static_cast<long long>(cfg.n_neq_steps) * static_cast<long long>(cfg.n_neq_traj) * 2LL;
        benchmark.n_force_evals_est = estimate_force_evals(benchmark.n_steps_total);
        benchmark.hyperparameters =
            "k=" + std::to_string(cfg.k) +
            ";k_midscale=" + std::to_string(cfg.k_midscale) +
            ";ntraj=" + std::to_string(cfg.n_neq_traj);
        append_lines(log_lines, benchmark_summary_lines(benchmark));
        write_log(resolve_log_path(opt.log_path), log_lines);
        std::cout << "Wrote NEQ outputs to " << cfg.out_dir << "\n";
        return 0;
    }

    print_usage();
    return 1;
}
