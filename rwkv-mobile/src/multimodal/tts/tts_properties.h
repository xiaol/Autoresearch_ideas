#pragma once

#include <map>
#include <string>
#include <vector>

namespace rwkvmobile {

const int tts_special_token_offset = 77823;

const std::map<std::string, int> SPEED_MAP = {
    {"very_slow", 1},
    {"slow", 2},
    {"medium", 3},
    {"fast", 4},
    {"very_fast", 5},
};

const std::map<std::string, int> PITCH_MAP = {
    {"low_pitch", 6},
    {"medium_pitch", 7},
    {"high_pitch", 8},
    {"very_high_pitch", 9},
};

const std::map<std::string, int> AGE_MAP = {
    {"child", 13},
    {"teenager", 14},
    {"youth-adult", 15},
    {"middle-aged", 16},
    {"elderly", 17},
};

const std::map<std::string, int> EMOTION_MAP = {
    {"UNKNOWN", 21},
    {"NEUTRAL", 22},
    {"ANGRY", 23},
    {"HAPPY", 24},
    {"SAD", 25},
    {"FEARFUL", 26},
    {"DISGUSTED", 27},
    {"SURPRISED", 28},
    {"SARCASTIC", 29},
    {"EXCITED", 30},
    {"SLEEPY", 31},
    {"CONFUSED", 32},
    {"EMPHASIS", 33},
    {"LAUGHING", 34},
    {"SINGING", 35},
    {"WORRIED", 36},
    {"WHISPER", 37},
    {"ANXIOUS", 38},
    {"NO-AGREEMENT", 39},
    {"APOLOGETIC", 40},
    {"CONCERNED", 41},
    {"ENUNCIATED", 42},
    {"ASSERTIVE", 43},
    {"ENCOURAGING", 44},
    {"CONTEMPT", 45},
};

const std::map<std::string, int> GENDER_MAP = {
    {"female", 46},
    {"male", 47}
};

std::vector<int> convert_standard_properties_to_tokens(
    std::string age,
    std::string gender,
    std::string emotion,
    std::string pitch,
    std::string speed
);

std::string classify_speed(float speed);
std::string classify_pitch(float pitch, std::string gender, std::string age);

} // namespace rwkvmobile