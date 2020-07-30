#include <iostream>

namespace std;

int main() {
    // 定义输入全连接层、隐藏层、输出层神经元个数
    int N = 60;
    int D_in = 1000;
    int H = 100;
    int D_out = 10;

    std::vector<int> v(1000);
    std::generate(v.begin(), v.end(), std::rand);   
}