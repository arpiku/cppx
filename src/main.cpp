#include "funcs.h"
#include "header.h"


int getSum(int a) {
    int sum = 0; for (int i = 0; i < a; i++)  {
        sum +=i;
    }
    return sum;
}

void foo() {
    std::cout << "asdf" << std::endl;
}


int productOfElements(const std::vector<int>& values) {
    int product = 1;
    for (int value : values) {
        product *= value;
    }
    return product;
}


int main() {
    lol<float> = new float();
    foo();
    int x = foo(5);

    std::vector<int> v1 = {0,2,3,5};
    std::vector<int> v2 = {2,5,8,5};

    int dd = getSum(50);
    std::cout << dd << std::endl;

    int dotProduct = dot_product(v1, v2);
    std::cout << dotProduct << std::endl;


    int cc = adder(2,3);
    std::cout << cc << std::endl;

    launch_kernel();


    return 0;
}
