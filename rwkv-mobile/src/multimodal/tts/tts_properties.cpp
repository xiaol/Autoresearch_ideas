#include "tts_properties.h"
#include "logger.h"
#include <algorithm>
#include <cctype>

namespace rwkvmobile {

static std::string lowercase(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return s;
}

static std::string uppercase(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return std::toupper(c);
    });
    return s;
}

std::vector<int> convert_standard_properties_to_tokens(
    std::string age,
    std::string gender,
    std::string emotion,
    std::string pitch,
    std::string speed
) {
    try {
        std::vector<int> tokens = {tts_special_token_offset};
        tokens.push_back(AGE_MAP.at(lowercase(age)));
        tokens.push_back(GENDER_MAP.at(lowercase(gender)));
        tokens.push_back(EMOTION_MAP.at(uppercase(emotion)));
        tokens.push_back(PITCH_MAP.at(lowercase(pitch)));
        tokens.push_back(SPEED_MAP.at(lowercase(speed)));
        return tokens;

    } catch (const std::exception& e) {
        LOGE("Error converting standard properties to tokens: %s", e.what());
        return std::vector<int>();
    }
}

std::vector<int> convert_properties_to_tokens(
    std::string age,
    std::string gender,
    std::string emotion,
    float pitch,
    float speed
) {
    try {
        std::vector<int> tokens = {tts_special_token_offset};
        tokens.push_back(AGE_MAP.at(lowercase(age)));
        tokens.push_back(GENDER_MAP.at(lowercase(gender)));
        tokens.push_back(EMOTION_MAP.at(uppercase(emotion)));
        tokens.push_back(PITCH_MAP.at(classify_pitch(pitch, gender, age)));
        tokens.push_back(SPEED_MAP.at(classify_speed(speed)));
        return tokens;

    } catch (const std::exception& e) {
        LOGE("Error converting properties to tokens: %s", e.what());
        return std::vector<int>();
    }
}

std::string classify_speed(float speed) {
    if (speed <= 3.5f) {
        return "very_slow";
    } else if (speed < 4.0f) {
        return "slow";
    } else if (speed <= 4.5f) {
        return "medium";
    } else if (speed <= 5.0f) {
        return "fast";
    } else {
        return "very_fast";
    }
}

std::string classify_pitch(float pitch, std::string gender, std::string age) {
    std::transform(gender.begin(), gender.end(), gender.begin(), ::tolower);
    std::transform(age.begin(), age.end(), age.begin(), ::tolower);

    if (gender == "female") {
        if (age == "child") {
            if (pitch < 250.0f) return "low_pitch";
            else if (pitch < 290.0f) return "medium_pitch";
            else return "high_pitch";
        } else if (age == "teenager") {
            if (pitch < 208.0f) return "low_pitch";
            else if (pitch < 238.0f) return "medium_pitch";
            else if (pitch < 270.0f) return "high_pitch";
            else return "very_high_pitch";
        } else if (age == "youth-adult") {
            if (pitch < 191.0f) return "low_pitch";
            else if (pitch < 211.0f) return "medium_pitch";
            else if (pitch < 232.0f) return "high_pitch";
            else return "very_high_pitch";
        } else if (age == "middle-aged") {
            if (pitch < 176.0f) return "low_pitch";
            else if (pitch < 195.0f) return "medium_pitch";
            else if (pitch < 215.0f) return "high_pitch";
            else return "very_high_pitch";
        } else if (age == "elderly") {
            if (pitch < 170.0f) return "low_pitch";
            else if (pitch < 190.0f) return "medium_pitch";
            else if (pitch < 213.0f) return "high_pitch";
            else return "very_high_pitch";
        } else {
            if (pitch < 187.0f) return "low_pitch";
            else if (pitch < 209.0f) return "medium_pitch";
            else if (pitch < 232.0f) return "high_pitch";
            else return "very_high_pitch";
        }
    } else if (gender == "male") {
        if (age == "teenager") {
            if (pitch < 121.0f) return "low_pitch";
            else if (pitch < 143.0f) return "medium_pitch";
            else if (pitch < 166.0f) return "high_pitch";
            else return "very_high_pitch";
        } else if (age == "youth-adult") {
            if (pitch < 115.0f) return "low_pitch";
            else if (pitch < 131.0f) return "medium_pitch";
            else if (pitch < 153.0f) return "high_pitch";
            else return "very_high_pitch";
        } else if (age == "middle-aged") {
            if (pitch < 110.0f) return "low_pitch";
            else if (pitch < 125.0f) return "medium_pitch";
            else if (pitch < 147.0f) return "high_pitch";
            else return "very_high_pitch";
        } else if (age == "elderly") {
            if (pitch < 115.0f) return "low_pitch";
            else if (pitch < 128.0f) return "medium_pitch";
            else if (pitch < 142.0f) return "high_pitch";
            else return "very_high_pitch";
        } else {
            if (pitch < 114.0f) return "low_pitch";
            else if (pitch < 130.0f) return "medium_pitch";
            else if (pitch < 151.0f) return "high_pitch";
            else return "very_high_pitch";
        }
    } else {
        if (pitch < 130.0f) return "low_pitch";
        else if (pitch < 180.0f) return "medium_pitch";
        else if (pitch < 220.0f) return "high_pitch";
        else return "very_high_pitch";
    }
}


} // namespace rwkvmobile