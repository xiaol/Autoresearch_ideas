#include <iostream>
#include "../src/multimodal/vision/clip.h"

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input> <output> <itype>" << std::endl;
        return 1;
    }

    clip_model_quantize(argv[1], argv[2], std::stoi(argv[3]));
    return 0;
}
