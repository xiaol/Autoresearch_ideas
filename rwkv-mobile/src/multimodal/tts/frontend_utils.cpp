#include <string>
#include <vector>
#include <functional>
#include <unordered_set>
#include <regex>
#include <cctype>
#include <sstream>
#include <iomanip>
#include <iostream>

#include "frontend_utils.h"

namespace rwkvmobile {

namespace tts_frontend_utils {

inline bool is_ascii(unsigned char c) {
    return c < 128;
}

bool contains_chinese(const std::string& text) {
    static const std::regex chinese_pattern("[\u4e00-\u9fff]");
    return std::regex_search(text, chinese_pattern);
}

std::string replace_corner_mark(const std::string& text) {
    std::string result = text;

    size_t pos = result.find("²");
    while (pos != std::string::npos) {
        result.replace(pos, 2, "平方");
        pos = result.find("²", pos + 4);
    }

    pos = result.find("³");
    while (pos != std::string::npos) {
        result.replace(pos, 2, "立方");
        pos = result.find("³", pos + 4);
    }
    return result;
}

std::string remove_bracket(const std::string& text) {
    std::string result = text;
    result = std::regex_replace(result, std::regex("（|）"), "");
    result = std::regex_replace(result, std::regex("【|】"), "");

    result = std::regex_replace(result, std::regex("`"), "");

    result = std::regex_replace(result, std::regex("——"), " ");
    return result;
}

std::string spell_out_number(const std::string& text,
                           const std::function<std::string(int)>& number_to_words) {
    std::string result;
    std::string current_number;

    for (size_t i = 0; i < text.length(); ++i) {
        if (std::isdigit(text[i])) {
            current_number += text[i];
        } else {
            if (!current_number.empty()) {
                result += number_to_words(std::stoi(current_number));
                current_number.clear();
            }
            result += text[i];
        }
    }

    // Handle number at the end of string
    if (!current_number.empty()) {
        result += number_to_words(std::stoi(current_number));
    }

    return result;
}

std::string replace_blank(const std::string& text) {
    std::string result;
    for (size_t i = 0; i < text.length(); ++i) {
        if (text[i] == ' ') {
            // Check if space is between ASCII characters
            if (i > 0 && i < text.length() - 1 &&
                is_ascii(text[i-1]) && text[i-1] != ' ' &&
                is_ascii(text[i+1]) && text[i+1] != ' ') {
                result += ' ';
            }
        } else {
            result += text[i];
        }
    }
    return result;
}

bool is_only_punctuation(const std::string& text) {
    if (text.empty()) return true;

    // Regular expression for punctuation marks
    // This includes both ASCII and Unicode punctuation
    static const std::regex punctuation_pattern("^[\\p{P}\\p{S}]*$");

    return std::regex_match(text, punctuation_pattern);
}

std::vector<std::string> split_paragraph(
    const std::string& text,
    const std::function<std::vector<int>(const std::string&)>& tokenize,
    const bool is_chinese,
    size_t token_max_n,
    size_t token_min_n,
    size_t merge_len,
    bool comma_split
) {
    // Lambda function to calculate utterance length
    auto calc_utt_length = [&tokenize, &is_chinese](const std::string& text) -> size_t {
        if (is_chinese) {
            size_t len = 0;
            for (size_t i = 0; i < text.length(); i++) {
                if ((text[i] & 0xC0) != 0x80) { // Count only non-continuation bytes in UTF-8
                    len++;
                }
            }
            return len;
        } else {
            return tokenize(text).size();
        }
    };

    // Lambda function to check if should merge
    auto should_merge = [&tokenize, &is_chinese, merge_len](const std::string& text) -> bool {
        if (is_chinese) {
            size_t len = 0;
            for (size_t i = 0; i < text.length(); i++) {
                if ((text[i] & 0xC0) != 0x80) { // Count only non-continuation bytes in UTF-8
                    len++;
                }
            }
            return len < merge_len;
        } else {
            return tokenize(text).size() < merge_len;
        }
    };

    // Set up punctuation marks
    std::unordered_set<std::string> pounc;
    if (is_chinese) {
        pounc = {"。", "？", "！", "；", "：", ".", "?", "!", ";"};
    } else {
        pounc = {".", "?", "!", ";"};
    }

    if (comma_split) {
        pounc.insert("，");
        pounc.insert(",");
    }

    // Add ending punctuation if not present
    std::string processed_text = text;
    if (is_chinese) {
        if (pounc.find(processed_text.substr(processed_text.length() - 3)) == pounc.end()) {
            processed_text += "。";
        }
    } else {
        if (pounc.find(processed_text.substr(processed_text.length() - 1)) == pounc.end()) {
            processed_text += ".";
        }
    }

    // Split into utterances
    std::vector<std::string> utts;
    size_t pos = 0;
    while (pos < processed_text.length()) {
        size_t next_pos = processed_text.length();
        for (const auto& p : pounc) {
            size_t found = processed_text.find(p, pos);
            if (found != std::string::npos && found < next_pos) {
                next_pos = found + p.length();
            }
        }

        if (next_pos > pos) {
            std::string utt = processed_text.substr(pos, next_pos - pos);
            if (!utt.empty()) {
                utts.push_back(utt);
            }
        }
        pos = next_pos;
    }

    // Merge utterances based on length constraints
    std::vector<std::string> final_utts;
    std::string cur_utt;

    for (const auto& utt : utts) {
        if (calc_utt_length(cur_utt + utt) > token_max_n && calc_utt_length(cur_utt) > token_min_n) {
            final_utts.push_back(cur_utt);
            cur_utt = "";
        }
        cur_utt += utt;
    }

    if (!cur_utt.empty()) {
        if (should_merge(cur_utt) && !final_utts.empty()) {
            final_utts.back() += cur_utt;
        } else {
            final_utts.push_back(cur_utt);
        }
    }

    return final_utts;
}

std::vector<std::string> process_text(
    const std::string& text,
    const std::function<std::vector<int>(const std::string&)>& tokenize,
#if !defined(_WIN32)
    std::vector<std::unique_ptr<kaldifst::TextNormalizer>> & tn_list_zh,
#endif
    size_t token_max_n,
    size_t token_min_n,
    size_t merge_len,
    bool comma_split
) {
    const bool is_chinese = contains_chinese(text);

    static auto replace_text = [](const std::string& text, const std::string& old_str, const std::string& new_str) -> std::string {
        std::string result = text;
        size_t pos = 0;
        while ((pos = result.find(old_str, pos)) != std::string::npos) {
            result.replace(pos, old_str.length(), new_str);
            pos += new_str.length();
        }
        return result;
    };

    std::string processed_text = text;
    if (is_chinese) {
        std::regex percentage_pattern("([0-9]+\\.?[0-9]*|π|e)%");
        processed_text = std::regex_replace(processed_text, percentage_pattern, "百分之$1");

#if !defined(_WIN32)
        if (!tn_list_zh.empty()) {
            for (const auto& tn : tn_list_zh) {
                processed_text = tn->Normalize(processed_text);
            }
        }
#endif

        processed_text = replace_text(processed_text, "\n", "");
        processed_text = replace_blank(processed_text);
        processed_text = replace_corner_mark(processed_text);
        processed_text = replace_text(processed_text, ".", "。");
        processed_text = replace_text(processed_text, " - ", "，");
        processed_text = remove_bracket(processed_text);
    } else {
        // TODO: add english inflect parser or use other method to spell out number
        // processed_text = spell_out_number(processed_text, inflect_parser);
        processed_text = replace_text(processed_text, "°F", " degrees Fahrenheit");
        processed_text = replace_text(processed_text, "°C", " degrees Celsius");
    }
    return split_paragraph(processed_text, tokenize, is_chinese, token_max_n, token_min_n, merge_len, comma_split);
}

} // namespace tts_frontend_utils

} // namespace rwkvmobile