// cling_executor.cpp
#include "cling_executor.hpp"
#include <sstream>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <regex>

#ifdef _WIN32
  #include <windows.h>
#endif

using std::string;
using std::vector;

namespace {
#ifdef HAVE_CLING
static std::vector<const char*> makeArgv(const std::vector<std::string>& args) {
    std::vector<const char*> argv;
    argv.reserve(args.size());
    for (auto& a : args) argv.push_back(a.c_str());
    return argv;
}
#endif

static bool is_intish(const json& v)    { return v.is_number_integer(); }
static bool is_uintish(const json& v)   { return v.is_number_unsigned(); }
static bool is_floatish(const json& v)  { return v.is_number_float(); }

} // namespace

bool ClingExecutor::initialize(const json& config) {
#ifdef HAVE_CLING
    if (initialized_) return true;

    extraArgs_.clear();
    extraArgs_.push_back("-std=c++17");
    extraArgs_.push_back("-O2");
    extraArgs_.push_back("-fno-rtti");
    extraArgs_.push_back("-Wno-unknown-warning-option");

    if (config.contains("includePaths") && config["includePaths"].is_array()) {
        for (auto& p : config["includePaths"]) {
            if (p.is_string()) extraArgs_.push_back(std::string("-I") + p.get<std::string>());
        }
    }
    if (config.contains("clangArgs") && config["clangArgs"].is_array()) {
        for (auto& a : config["clangArgs"]) {
            if (a.is_string()) extraArgs_.push_back(a.get<std::string>());
        }
    }

    auto argv = makeArgv(extraArgs_);
    try {
        interp_ = std::make_unique<cling::Interpreter>(static_cast<int>(argv.size()), argv.data());
    } catch (const std::exception& e) {
        std::cerr << "[cling] Interpreter init failed: " << e.what() << std::endl;
        return false;
    }

    initialized_ = true;
    return true;
#else
    (void)config;
    std::cerr << "ClingExecutor built without HAVE_CLING. Not available.\n";
    return false;
#endif
}

void ClingExecutor::cleanup() {
#ifdef HAVE_CLING
    interp_.reset();
#endif
    initialized_ = false;
}

static inline std::string toLowerCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return char(std::tolower(c)); });
    return s;
}

std::string ClingExecutor::sanitizeIdent(const std::string& s) {
    if (s.empty()) return "_u";
    std::string out;
    out.reserve(s.size());
    auto validFirst = [](char c){
        return std::isalpha(static_cast<unsigned char>(c)) || c=='_';
    };
    auto valid = [](char c){
        return std::isalnum(static_cast<unsigned char>(c)) || c=='_';
    };
    for (size_t i=0;i<s.size();++i) {
        char c = s[i];
        if ((i==0 && !validFirst(c)) || (i>0 && !valid(c))) c = '_';
        out.push_back(c);
    }
    static const char* kws[] = {"int","float","double","auto","class","struct",
        "union","void","char","short","long","signed","unsigned",
        "if","else","for","while","do","switch","case","default",
        "break","continue","return"};
    for (auto kw : kws) {
        if (out == kw) { out += "_u"; break; }
    }
    return out;
}

bool ClingExecutor::buildUniformList(const TaskData& task, std::vector<UniformValue>& uniforms) const {
    uniforms.clear();

    auto pushUnique = [&](const std::string& name, UniformValue::Type t, const json& v) {
        auto it = std::find_if(uniforms.begin(), uniforms.end(), [&](const UniformValue& u){ return u.name==name; });
        if (it != uniforms.end()) uniforms.erase(it);
        UniformValue u; u.name = name; u.type = t;
        if (t==UniformValue::Type::INT32)   u.i32 = v.get<int32_t>();
        if (t==UniformValue::Type::UINT32)  u.u32 = v.get<uint32_t>();
        if (t==UniformValue::Type::FLOAT32) u.f32 = v.get<float>();
        uniforms.push_back(u);
    };

    if (task.metadata.contains("schema") && task.metadata["schema"].contains("uniforms")
        && task.metadata["schema"]["uniforms"].is_array())
    {
        for (const auto& def : task.metadata["schema"]["uniforms"]) {
            if (!def.contains("name") || !def.contains("type")) continue;
            const std::string name = def["name"].get<std::string>();
            const std::string type = toLowerCopy(def["type"].get<std::string>());
            const json* src = nullptr;

            if (task.metadata.contains(name)) src = &task.metadata[name];
            else if (task.metadata.contains("uniforms") && task.metadata["uniforms"].contains(name))
                src = &task.metadata["uniforms"][name];
            else if (task.chunkUniforms.contains(name)) src = &task.chunkUniforms[name];

            if (!src || src->is_null()) continue;

            if ((type=="i32" || type=="int" || type=="int32") && (is_intish(*src) || is_uintish(*src))) {
                pushUnique(name, UniformValue::Type::INT32, *src);
            } else if ((type=="u32" || type=="uint" || type=="uint32") && (is_uintish(*src) || is_intish(*src))) {
                pushUnique(name, UniformValue::Type::UINT32, *src);
            } else if ((type=="f32" || type=="float" || type=="float32") &&
                       (is_floatish(*src) || is_intish(*src) || is_uintish(*src))) {
                json j = *src;
                if (j.is_number_integer() || j.is_number_unsigned()) j = j.get<double>();
                pushUnique(name, UniformValue::Type::FLOAT32, j);
            }
        }
    }

    std::vector<std::string> extraKeys;
    for (auto it = task.metadata.begin(); it != task.metadata.end(); ++it) {
        if (it.key()=="schema" || it.key()=="uniforms") continue;
        if (is_intish(it.value()) || is_uintish(it.value()) || is_floatish(it.value()))
            extraKeys.push_back(it.key());
    }
    std::sort(extraKeys.begin(), extraKeys.end());
    for (const auto& k : extraKeys) {
        const auto& v = task.metadata[k];
        if (is_intish(v))    pushUnique(k, UniformValue::Type::INT32, v);
        else if (is_uintish(v)) pushUnique(k, UniformValue::Type::UINT32, v);
        else if (is_floatish(v)) pushUnique(k, UniformValue::Type::FLOAT32, v);
    }

    if (!task.chunkUniforms.empty()) {
        for (auto it = task.chunkUniforms.begin(); it != task.chunkUniforms.end(); ++it) {
            const auto& k = it.key();
            const auto& v = it.value();
            if (is_intish(v))    pushUnique(k, UniformValue::Type::INT32, v);
            else if (is_uintish(v)) pushUnique(k, UniformValue::Type::UINT32, v);
            else if (is_floatish(v)) pushUnique(k, UniformValue::Type::FLOAT32, v);
        }
    }

    return true;
}

std::string ClingExecutor::generatePrelude(const std::string& entry,
                                           const std::vector<UniformValue>& uniforms) const {
    std::ostringstream ss;
    ss << R"CPP(
// ====== BEGIN: executor prelude (generated) ======
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <cstring>
extern "C" {
)CPP";

    ss << "struct __uniforms_t {\n";
    for (const auto& u : uniforms) {
        std::string field = sanitizeIdent(u.name);
        switch (u.type) {
            case UniformValue::Type::INT32:   ss << "  int32_t "  << field << ";\n"; break;
            case UniformValue::Type::UINT32:  ss << "  uint32_t " << field << ";\n"; break;
            case UniformValue::Type::FLOAT32: ss << "  float "    << field << ";\n"; break;
        }
    }
    ss << "};\n\n";

    ss << "extern \"C\" size_t __uniforms_sizeof() { return sizeof(__uniforms_t); }\n";
    ss << "extern \"C\" void __write_uniforms(void* dst) {\n";
    ss << "  __uniforms_t tmp{};\n";
    for (const auto& u : uniforms) {
        std::string field = sanitizeIdent(u.name);
        switch (u.type) {
            case UniformValue::Type::INT32:   ss << "  tmp." << field << " = " << u.i32 << ";\n"; break;
            case UniformValue::Type::UINT32:  ss << "  tmp." << field << " = " << u.u32 << "u;\n"; break;
            case UniformValue::Type::FLOAT32: {
                std::ostringstream f; f.setf(std::ios::fixed); f<<std::setprecision(9)<<u.f32;
                ss << "  tmp." << field << " = " << f.str() << "f;\n"; break;
            }
        }
    }
    ss << "  std::memcpy(dst, &tmp, sizeof(__uniforms_t));\n";
    ss << "}\n\n";

    ss << "extern void " << entry << "(size_t idx, void** inputs, void** outputs, const __uniforms_t& u);\n";
    ss << "extern \"C\" void __kernel_shim(size_t idx, void** inputs, void** outputs, const void* uniforms) {\n";
    ss << "  const __uniforms_t& u = *reinterpret_cast<const __uniforms_t*>(uniforms);\n";
    ss << "  " << entry << "(idx, inputs, outputs, u);\n";
    ss << "}\n";
    ss << "extern \"C\" void* __get_kernel_shim_ptr() { return (void*)&__kernel_shim; }\n";
    ss << "} // extern \"C\"\n";
    ss << "// ======  END: executor prelude (generated) ======\n\n";
    return ss.str();
}

size_t ClingExecutor::resolveDispatchCount(const TaskData& task) const {
    auto getNum = [&](const char* key)->size_t{
        if (task.metadata.contains(key) && task.metadata[key].is_number()) {
            return task.metadata[key].get<size_t>();
        }
        if (task.chunkUniforms.contains(key) && task.chunkUniforms[key].is_number()) {
            return task.chunkUniforms[key].get<size_t>();
        }
        return 0;
    };
    size_t N = getNum("N");
    if (!N) N = getNum("elements");
    if (!N) N = getNum("elementCount");
    if (!N && !task.outputSizes.empty()) N = task.outputSizes[0];
    if (!N) N = 1;
    return N;
}

TaskResult ClingExecutor::executeTask(const TaskData& task) {
    TaskResult result;
    result.success = false;

#ifndef HAVE_CLING
    result.errorMessage = "Cling not available (build without HAVE_CLING)";
    return result;
#else
    if (!initialized_) {
        result.errorMessage = "ClingExecutor not initialized";
        return result;
    }

    const auto t0 = std::chrono::high_resolution_clock::now();

    try {
        std::vector<UniformValue> uniforms;
        buildUniformList(task, uniforms);
        const std::string prelude = generatePrelude(task.entry, uniforms);

        std::string tu;
        tu.reserve(prelude.size() + task.kernel.size() + 32);
        tu += prelude;
        tu += "\n// ====== BEGIN: user kernel ======\n";
        tu += task.kernel;
        tu += "\n// ======  END: user kernel ======\n";

        if (auto rc = interp_->process(tu)) {
            std::ostringstream msg;
            msg << "Cling compile failed (rc=" << rc << ")";
            result.errorMessage = msg.str();
            return result;
        }

        cling::Value vShim, vUSize, vUWrite;
        if (interp_->evaluate("(void*)__get_kernel_shim_ptr()", vShim)) {
            result.errorMessage = "Failed to resolve __get_kernel_shim_ptr()";
            return result;
        }
        if (interp_->evaluate("(size_t)__uniforms_sizeof()", vUSize)) {
            result.errorMessage = "Failed to resolve __uniforms_sizeof()";
            return result;
        }
        if (interp_->evaluate("(void*)__write_uniforms", vUWrite)) {
            result.errorMessage = "Failed to resolve __write_uniforms";
            return result;
        }

        using KernelFn = void(*)(size_t, void**, void**, const void*);
        using WriteUniformsFn = void(*)(void*);

        KernelFn kernel = reinterpret_cast<KernelFn>(vShim.getAs<void*>());
        size_t uSize    = static_cast<size_t>(vUSize.getAs<size_t>());
        WriteUniformsFn writeUniforms = reinterpret_cast<WriteUniformsFn>(vUWrite.getAs<void*>());

        if (!kernel || !writeUniforms) {
            result.errorMessage = "Resolved null function pointer(s) from Cling TU";
            return result;
        }

        const size_t inCount  = task.inputData.size();
        const size_t outCount = task.outputSizes.size();

        vector<void*> inputPtrs; inputPtrs.reserve(inCount);
        for (const auto& buf : task.inputData) {
            inputPtrs.push_back(const_cast<uint8_t*>(buf.data()));
        }

        vector<vector<uint8_t>> outputs(outCount);
        vector<void*> outputPtrs; outputPtrs.reserve(outCount);
        for (size_t i=0;i<outCount;++i) {
            outputs[i].resize(task.outputSizes[i], 0);
            outputPtrs.push_back(outputs[i].data());
        }

        std::unique_ptr<uint8_t[]> uniformsBlob;
        if (uSize) {
            uniformsBlob.reset(new uint8_t[uSize]);
            std::memset(uniformsBlob.get(), 0, uSize);
            writeUniforms(uniformsBlob.get());
        }

        size_t N = resolveDispatchCount(task);
        for (size_t i=0;i<N;++i) {
            kernel(i, inputPtrs.data(), outputPtrs.data(), (const void*)uniformsBlob.get());
        }

        result.outputData = std::move(outputs);
        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("ClingExecutor exception: ") + e.what();
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    result.processingTime = (double)std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() / 1000.0;
    return result;
#endif
}

json ClingExecutor::getCapabilities() const {
    json caps = {
        {"framework", getFrameworkName()},
        {"executionModel", "single_thread_loop"},
        {"language", "c++17"},
        {"jit", true},
        {"supportsUniforms", true},
        {"supportsMultipleInputs", true},
        {"supportsMultipleOutputs", true}
    };
#ifndef HAVE_CLING
    caps["available"] = false;
    caps["reason"] = "built without HAVE_CLING";
#else
    caps["available"] = initialized_;
#endif
    return caps;
}
