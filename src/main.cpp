#include "bencher.h"
#include <iostream>
#include <ranges>

const uint32_t SEED = 54;

int main() {
    Bencher::TestCase t = Bencher::generate_square_case(100,SEED);

    return 0;
}
