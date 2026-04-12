#ifndef RWKVMOBILE_TTS_FRONTEND_UTILS_H
#define RWKVMOBILE_TTS_FRONTEND_UTILS_H

#include <string>
#include <vector>
#include <functional>

#if !defined(_WIN32)
#include "kaldifst/csrc/text-normalizer.h"
#endif

namespace rwkvmobile {

namespace tts_frontend_utils {

inline bool is_ascii(unsigned char c);

bool contains_chinese(const std::string& text);

std::string replace_corner_mark(const std::string& text);

std::string remove_bracket(const std::string& text);

std::string spell_out_number(const std::string& text,
                           const std::function<std::string(int)>& number_to_words);

std::string replace_blank(const std::string& text);

bool is_only_punctuation(const std::string& text);

std::vector<std::string> split_paragraph(
    const std::string& text,
    const std::function<std::vector<int>(const std::string&)>& tokenize,
    const bool is_chinese,
    size_t token_max_n = 40,
    size_t token_min_n = 20,
    size_t merge_len = 10,
    bool comma_split = false
);

std::vector<std::string> process_text(
    const std::string& text,
    const std::function<std::vector<int>(const std::string&)>& tokenize,
#if !defined(_WIN32)
    std::vector<std::unique_ptr<kaldifst::TextNormalizer>> & tn_list_zh,
#endif
    size_t token_max_n = 40,
    size_t token_min_n = 20,
    size_t merge_len = 10,
    bool comma_split = false
);

} // namespace tts_frontend_utils

} // namespace rwkvmobile

#endif // RWKVMOBILE_TTS_FRONTEND_UTILS_H