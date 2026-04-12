#include "backend.h"
#include "mlx_rwkv_backend.h"
#include "commondef.h"
#include "logger.h"
#include "MLXModelFFI.h"
#include <filesystem>

namespace rwkvmobile {

static bool mlx_initialized = false;

int mlx_rwkv_backend::init(void * extra) {
    if (!mlx_initialized) {
        if (mlx_initialize() != 0) {
            LOGE("Failed to initialize MLX\n");
            return RWKV_ERROR_INIT;
        }
        mlx_initialized = true;
    }

    LOGI("MLX initialized successfully\n");
    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::load_model(std::string model_path, void * extra) {
    namespace fs = std::filesystem;

    std::error_code ec;
    bool exists = fs::exists(model_path, ec);
    if (ec || !exists) {
        LOGE("MLX model path does not exist: %s\n", model_path.c_str());
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }

    if (!fs::is_directory(model_path, ec)) {
        LOGE(
            "MLX backend expects a model directory, but got a file: %s\n"
            "The bundled MLXModelFFI looks for files such as config.json inside the model path.\n"
            "A single .st / .safetensors RWKV checkpoint is not sufficient for this backend.\n",
            model_path.c_str()
        );
        return RWKV_ERROR_MODEL | RWKV_ERROR_INVALID_PARAMETERS;
    }

    fs::path config_path = fs::path(model_path) / "config.json";
    if (!fs::exists(config_path, ec)) {
        LOGE(
            "MLX model directory is missing config.json: %s\n"
            "This MLX backend appears to expect an MLX-LM style model folder rather than a raw RWKV checkpoint file.\n",
            config_path.string().c_str()
        );
        return RWKV_ERROR_MODEL | RWKV_ERROR_INVALID_PARAMETERS;
    }

    if (!model_handle) {
        model_handle = mlx_model_load(model_path.c_str());
        if (!model_handle) {
            LOGE("Failed to load MLX model: %s\n", mlx_last_error_message());
            return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
        }
    }

    int vocab_size, hidden_size, head_dim, num_layers;
    if (mlx_model_get_config(model_handle, &vocab_size, &hidden_size, &head_dim, &num_layers) != 0) {
        LOGE("Failed to get MLX model config\n");
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }

    this->vocab_size = vocab_size;
    this->n_layers = num_layers;
    this->num_heads = head_dim;
    this->hidden_size = hidden_size;

    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::eval(int id, Tensor1D & logits) {
    if (!model_handle) {
        LOGE("MLX model not loaded\n");
        return RWKV_ERROR_EVAL;
    }

    if (logits_buffer.size() != vocab_size) {
        logits_buffer.resize(vocab_size);
    }

    int ret = mlx_model_eval(model_handle, &id, 1, logits_buffer.data());
    if (ret != 0) {
        LOGE("Failed to evaluate MLX model: %s\n", mlx_last_error_message());
        return RWKV_ERROR_EVAL;
    }

    logits = Tensor1D::make(logits_buffer.data(), TensorDType::F32, (size_t)vocab_size);
    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::eval(std::vector<int> ids, Tensor1D & logits) {
    if (!model_handle) {
        LOGE("MLX model not loaded\n");
        return RWKV_ERROR_EVAL;
    }

    if (logits_buffer.size() != vocab_size) {
        logits_buffer.resize(vocab_size);
    }

    std::vector<int32_t> ids_u32(ids.begin(), ids.end());
    int ret = mlx_model_eval(model_handle, ids_u32.data(), ids_u32.size(), logits_buffer.data());
    if (ret != 0) {
        LOGE("Failed to evaluate MLX model: %s\n", mlx_last_error_message());
        return RWKV_ERROR_EVAL;
    }

    logits = Tensor1D::make(logits_buffer.data(), TensorDType::F32, (size_t)vocab_size);
    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::get_state(std::any &state) {
    if (!model_handle) {
        LOGE("MLX model not loaded\n");
        return RWKV_ERROR_EVAL;
    }

    int32_t cache_size = n_layers * (num_heads + 2) * hidden_size * sizeof(__fp16);

    std::vector<__fp16> cache_buffer(cache_size / sizeof(__fp16), 0.0f);
    mlx_cache_read(model_handle, (void*)cache_buffer.data(), cache_size);
    state = std::move(cache_buffer);

    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::set_state(std::any state) {
    if (!model_handle) {
        LOGE("MLX model not loaded\n");
        return RWKV_ERROR_EVAL;
    }

    std::vector<__fp16> cache_buffer = std::any_cast<std::vector<__fp16>>(state);
    int ret = mlx_cache_write(model_handle, (const void*)cache_buffer.data(), cache_buffer.size() * sizeof(__fp16));
    if (ret != 0) {
        LOGE("Failed to write MLX cache\n");
        return RWKV_ERROR_EVAL;
    }

    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::free_state(std::any state) {
    state.reset();

    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::zero_state() {
    if (!model_handle) {
        LOGE("MLX model not loaded\n");
        return RWKV_ERROR_EVAL;
    }

    int32_t cache_size = n_layers * (num_heads + 2) * hidden_size * sizeof(__fp16);
    std::vector<__fp16> cache_buffer(cache_size / sizeof(__fp16), 0.0f);
    int ret = mlx_cache_write(model_handle, (const void*)cache_buffer.data(), cache_size);
    if (ret != 0) {
        LOGE("Failed to zero MLX cache: %s\n", mlx_last_error_message() == NULL ? "Unknown error" : mlx_last_error_message());
        return RWKV_ERROR_EVAL;
    }

    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::release_model() {
    if (model_handle) {
        mlx_model_release(model_handle);
        model_handle = NULL;
    }
    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::release() {
    return RWKV_SUCCESS;
}

bool mlx_rwkv_backend::is_available() {
    return true;
}

} // namespace rwkvmobile
