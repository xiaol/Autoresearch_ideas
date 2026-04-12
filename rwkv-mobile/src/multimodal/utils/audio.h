#pragma once

#include <vector>
#include <string>
#include <cstdint>
namespace rwkvmobile {

class wav_file {
public:
    wav_file() {
        sample_rate = 0;
        num_channels = 0;
        bit_depth = 0;
        num_samples = 0;
    };
    ~wav_file() = default;
    bool load(const std::string& path);
    bool save(const std::string& path);

    void resample(int new_sample_rate);

    std::vector<float> samples;

    int16_t audio_format;
    int16_t num_channels;
    int32_t sample_rate;
    int32_t byte_rate;
    int16_t block_align;
    int16_t bit_depth;
    int32_t num_samples;
};

std::vector<std::vector<float>> melSpectrogram(std::vector<float>& audio, int sample_rate, int n_fft, int n_hop, int n_mel, int fmin, int fmax, float power, bool center, bool return_magnitude);
std::vector<std::vector<float>> logMelSpectrogram(std::vector<float>& audio, int sample_rate, int n_fft, int n_hop, int n_mel, int fmin, int fmax, float power, bool center, bool return_magnitude);

void dynamic_range_compression(std::vector<std::vector<float>>& features);

void audio_volume_normalize(std::vector<float>& audio, float coeff = 0.2f);

void save_samples_to_wav(std::vector<float> samples, std::string path, int sample_rate = 24000);

}