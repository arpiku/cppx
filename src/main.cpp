#include <numeric>
#include <vector>
#include <iostream>
#include <algorithm>


int main() {
    std::vector<int> vec(100);
    std::iota(vec.rbegin(), vec.rend(), 0);

    for (int i : vec) {
        std::cout << i << " ";
    }

    std::cout << " ---------- \n";
    std::reverse(vec.begin(), vec.begin() + 50);

    for (int i : vec) {
        std::cout << i << " ";
    }

    return 0;
}
