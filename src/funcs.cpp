#include "funcs.h"

int dot_product(std::vector<int> a, std::vector<int> b) {
    if(a.size() != b.size()) {
        return 0;
    }
    int result = 0;
    for(int i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}

int adder(int a, int b) {
    return a + b;
}
