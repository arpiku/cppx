#include "funcs.h"

void longprinter() {
   for(int i = 0; i < 100; i++) {
       std::cout << "___________________________" << std::endl;
   }
}

int dot_roduct(std::vector<int> a, std::vector<int> b) {
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
