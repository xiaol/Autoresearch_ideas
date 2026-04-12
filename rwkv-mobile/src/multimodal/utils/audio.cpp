#include "audio.h"
#include "logger.h"
#include "librosa.h"
#include <fstream>
#include <cstdint>
#include <cmath>
#include <cstring>

namespace rwkvmobile {

static int16_t twoBytesToInt16(const char* bytes) {
    if (bytes == nullptr) {
        return 0;
    }
    return reinterpret_cast<const int16_t*>(bytes)[0];
}

static int32_t fourBytesToInt32(const char* bytes) {
    if (bytes == nullptr) {
        return 0;
    }
    return reinterpret_cast<const int32_t*>(bytes)[0];
}

bool wav_file::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    char info[5] = {0};
    file.read(info, 4);
    if (info[0] != 'R' || info[1] != 'I' || info[2] != 'F' || info[3] != 'F') {
        return false;
    }

    file.read(info, 4); // chunk size
    file.read(info, 4); // WAVE
    if (info[0] != 'W' || info[1] != 'A' || info[2] != 'V' || info[3] != 'E') {
        return false;
    }
    file.read(info, 4); // fmt
    file.read(info, 4); // fmt size
    file.read(info, 2); // audio format
    audio_format = twoBytesToInt16(info);
    LOGI("[WAV] audio_format: %d", audio_format);
    file.read(info, 2); // num channels
    num_channels = twoBytesToInt16(info);
    LOGI("[WAV] num_channels: %d", num_channels);

    file.read(info, 4); // sample rate
    sample_rate = fourBytesToInt32(info);
    LOGI("[WAV] sample_rate: %d", sample_rate);

    file.read(info, 4); // byte rate
    byte_rate = fourBytesToInt32(info);
    LOGI("[WAV] byte_rate: %d", byte_rate);

    file.read(info, 2); // block align
    block_align = twoBytesToInt16(info);
    LOGI("[WAV] block_align: %d", block_align);

    file.read(info, 2); // bit depth
    bit_depth = twoBytesToInt16(info);
    LOGI("[WAV] bit_depth: %d", bit_depth);

    file.read(info, 4); // chunk name
    std::string chunk_name(info, 4);
    while (chunk_name != "data") {
        file.read(info, 4); // chunk size
        int32_t chunk_size = fourBytesToInt32(info);
        for (int32_t i = 0; i < chunk_size / 2; i++) {
            file.read(info, 2);
        }
        file.read(info, 4); // data format
        chunk_name = std::string(info, 4);
    }
    file.read(info, 4); // data size
    num_samples = fourBytesToInt32(info) / (bit_depth / 8);
    LOGI("[WAV] num_samples: %d", num_samples);

    if (bit_depth == 16) {
        std::vector<int16_t> samples_int16(num_samples);
        file.read(reinterpret_cast<char*>(samples_int16.data()), num_samples * sizeof(int16_t));
        samples.resize(num_samples);
        for (int i = 0; i < num_samples; i++) {
            samples[i] = static_cast<float>(samples_int16[i]) / 32768.0f;
        }
    } else if (bit_depth == 8) {
        std::vector<int8_t> samples_int8(num_samples);
        file.read(reinterpret_cast<char*>(samples_int8.data()), num_samples * sizeof(int8_t));
        samples.resize(num_samples);
        for (int i = 0; i < num_samples; i++) {
            samples[i] = static_cast<float>(samples_int8[i]) / 128.0f;
        }
    } else if (bit_depth == 24) {
        std::vector<int8_t> data_int24(num_samples * 3);
        file.read(reinterpret_cast<char*>(data_int24.data()), num_samples * 3 * sizeof(int8_t));
        samples.resize(num_samples);
        for (int i = 0; i < num_samples; i++) {
            int32_t value = 0;
            memcpy(&value, data_int24.data() + i * 3, 3);

            if (value & 0x800000) {
                value |= 0xFF000000;
            }
            samples[i] = static_cast<float>(value) / 8388608.0f;
        }
    } else {
        LOGE("[WAV] Unsupported bit depth yet: %d", bit_depth);
        return false;
    }
    file.close();

    return true;
}

bool wav_file::save(const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.write("RIFF", 4);
    int32_t chunk_size = num_samples * (bit_depth / 8) + 36;
    file.write(reinterpret_cast<const char*>(&chunk_size), 4);
    file.write("WAVE", 4);
    file.write("fmt ", 4);
    int32_t fmt_size = 16;
    file.write(reinterpret_cast<const char*>(&fmt_size), 4);
    file.write(reinterpret_cast<const char*>(&audio_format), 2);
    file.write(reinterpret_cast<const char*>(&num_channels), 2);
    file.write(reinterpret_cast<const char*>(&sample_rate), 4);
    file.write(reinterpret_cast<const char*>(&byte_rate), 4);
    file.write(reinterpret_cast<const char*>(&block_align), 2);
    file.write(reinterpret_cast<const char*>(&bit_depth), 2);
    file.write("data", 4);
    int32_t data_size = num_samples * (bit_depth / 8);
    file.write(reinterpret_cast<const char*>(&data_size), 4);
    for (int i = 0; i < num_samples; i++) {
        if (bit_depth == 16) {
            int16_t sample = static_cast<int16_t>(samples[i] * 32768.0f);
            file.write(reinterpret_cast<const char*>(&sample), 2);
        } else if (bit_depth == 8) {
            int8_t sample = static_cast<int8_t>(samples[i] * 128.0f);
            file.write(reinterpret_cast<const char*>(&sample), 1);
        }
    }
    file.close();
    return true;
}

void wav_file::resample(int new_sample_rate) {
    if (samples.empty()) {
        LOGE("[WAV] samples is empty");
        return;
    }
    if (sample_rate == new_sample_rate) {
        return;
    }
    LOGI("[WAV] resampling from %d to %d", sample_rate, new_sample_rate);
    LOGD("[WAV] origin num_samples: %d", num_samples);

    int new_num_samples = (int)(num_samples * (float)new_sample_rate / (float)sample_rate);
    LOGD("[WAV] new num_samples: %d", new_num_samples);

    std::vector<float> resampled_samples(new_num_samples);

    // Linear interpolation resampling
    for (int i = 0; i < new_num_samples; i++) {
        float pos = (float)i * sample_rate / new_sample_rate;
        int pos0 = (int)pos;
        int pos1 = pos0 + 1;
        float frac = pos - pos0;

        if (pos1 >= num_samples) {
            resampled_samples[i] = samples[pos0];
        } else {
            resampled_samples[i] = samples[pos0] * (1.0f - frac) + samples[pos1] * frac;
        }
    }

    samples = resampled_samples;
    sample_rate = new_sample_rate;
    num_samples = new_num_samples;
}

std::vector<std::vector<float>> melSpectrogram(std::vector<float>& audio, int sample_rate, int n_fft, int n_hop, int n_mel, int fmin, int fmax, float power, bool center, bool return_magnitude) {
    return librosa::Feature::melspectrogram(audio, sample_rate, n_fft, n_hop, "hann", center, "reflect", power, n_mel, fmin, fmax, return_magnitude);
}

std::vector<std::vector<float>> logMelSpectrogram(std::vector<float>& audio, int sample_rate, int n_fft, int n_hop, int n_mel, int fmin, int fmax, float power, bool center, bool return_magnitude) {
    std::vector<std::vector<float>> mels = melSpectrogram(audio, sample_rate, n_fft, n_hop, n_mel, fmin, fmax, power, center, return_magnitude);

    float max_val = -1e20;
    for (int i = 0; i < mels.size(); i++) {
        for (int j = 0; j < mels[i].size(); j++) {
            mels[i][j] = log10f(std::max(mels[i][j], 1e-10f));
            max_val = std::max(max_val, mels[i][j]);
        }
    }

    for (int i = 0; i < mels.size(); i++) {
        for (int j = 0; j < mels[i].size(); j++) {
            mels[i][j] = std::max(mels[i][j], max_val - 8.0f);
            mels[i][j] = (mels[i][j] + 4.0f) / 4.0f;
        }
    }
    return mels;
}

void dynamic_range_compression(std::vector<std::vector<float>>& features) {
    for (int i = 0; i < features.size(); i++) {
        for (int j = 0; j < features[i].size(); j++) {
            features[i][j] = log(std::max(1e-5f, features[i][j]));
        }
    }
}

void audio_volume_normalize(std::vector<float>& audio, float coeff) {
    if (audio.empty()) {
        return;
    }

    std::vector<float> abs_sorted_audio(audio);
    std::sort(abs_sorted_audio.begin(), abs_sorted_audio.end(), [](float a, float b) {
        return std::abs(a) < std::abs(b);
    });
    float abs_max_val = abs_sorted_audio[abs_sorted_audio.size() - 1];

    if (abs_max_val < 0.1f) {
        float scale = abs_max_val > 1e-3f ? abs_max_val : 1e-3f;
        for (int i = 0; i < audio.size(); i++) {
            audio[i] = audio[i] / scale * 0.1f;
        }
    }

    std::vector<float> temp;
    for (int i = 0; i < abs_sorted_audio.size(); i++) {
        if (abs_sorted_audio[i] > 0.01f) {
            temp.push_back(abs_sorted_audio[i]);
        }
    }

    if (temp.size() <= 10) {
        return;
    }

    float volume = 0.0f;
    for (int i = (int)(temp.size() * 0.9f); i < (int)(temp.size() * 0.99f); i++) {
        volume += temp[i];
    }
    volume /= ((int)(temp.size() * 0.99f) - (int)(temp.size() * 0.9f));

    float scale = std::max(0.1f, std::min(10.0f, coeff / volume));
    for (int i = 0; i < audio.size(); i++) {
        audio[i] = audio[i] * scale;
    }

    abs_max_val = *std::max_element(audio.begin(), audio.end(), [](float a, float b) {
        return std::abs(a) < std::abs(b);
    });

    if (abs_max_val > 1.0f) {
        for (int i = 0; i < audio.size(); i++) {
            audio[i] = audio[i] / abs_max_val;
        }
    }

    return;
}

void save_samples_to_wav(std::vector<float> samples, std::string path, int sample_rate) {
    wav_file wav_file;
    wav_file.sample_rate = sample_rate;
    wav_file.num_channels = 1;
    wav_file.num_samples = samples.size();
    wav_file.bit_depth = 16;
    wav_file.audio_format = 1;
    wav_file.byte_rate = sample_rate * 16 / 8;
    wav_file.block_align = 2;
    wav_file.samples = samples;
    wav_file.save(path);
}

}
