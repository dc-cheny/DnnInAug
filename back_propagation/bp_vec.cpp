#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <functional>
#include <cmath> // Needed for the pow function
using namespace std;


vector<vector<double>> generate_randmat(int d1, int d2);
vector<vector<double>> vec_dot(vector<vector<double>> v1, vector<vector<double>> v2);
double mse_loss(vector<vector<double>> v1, vector<vector<double>> v2);
vector<vector<double>> const_dot(vector<vector<double>> v1, double c);
vector<vector<double>> vec_minus(vector<vector<double>> v1, vector<vector<double>> v2);
vector<vector<double>> vec_transpose(vector<vector<double>> v);


int main() {
    // 定义输入全连接层、隐藏层、输出层神经元个数
    int N = 60;
    int D_in = 1000;
    int H = 100;
    int D_out = 10;
    // 定义学习率
    double learning_rate = 1e-6;

    vector<vector<double>> x = generate_randmat(N, D_in); // 60 * 1000
    vector<vector<double>> y = generate_randmat(N, D_out); // 60 * 10
    vector<vector<double>> w1 = generate_randmat(D_in, H); // 1000 * 100
    vector<vector<double>> w2 = generate_randmat(H, D_out); // 100 * 10
    
    // std::cout << r[0][0] << endl;
    for (int iter=0;iter<5000;iter++) {
        // forward propagation
        vector<vector<double>> h = vec_dot(x, w1);
        // relu
        for (int _i=0;_i<h.size();_i++) {
            for (int _j=0;_j<h[0].size();_j++) {
                if (h[_i][_j] < 0) {
                    h[_i][_j] = 0;
                }
            }
        }
        vector<vector<double>> y_pred = vec_dot(h, w2);
        // calc loss
        double loss = mse_loss(y_pred, y);
        if (iter % 500 == 0) {
            cout << "第" << iter << "轮：loss为" << loss << endl;
        }
        // backward propagation
        vector<vector<double>> y_pred_grad = const_dot(vec_minus(y_pred, y), 2);
        vector<vector<double>> w2_grad = vec_dot(vec_transpose(h), y_pred_grad);
        vector<vector<double>> h_relu_grad = vec_dot(y_pred_grad, vec_transpose(w2));
        
    }
    
    
}

// 生成随机矩阵
vector<vector<double>> generate_randmat(int d1, int d2) 
{
    vector<int> init_v(d1 * d2);
    generate(init_v.begin(), init_v.end(), rand);
    std::vector<vector<double>> v(d1, vector<double>(d2));

    for(int i=0;i<d1;i++){//初始化
         for(int j=0;j<d2;j++){
            v[i][j] = init_v[j+d2*i]%100/(double)101;
         }
    }

    return v; // 返回二维矩阵
}

// 二维矩阵点乘
vector<vector<double>> vec_dot(vector<vector<double>> v1, vector<vector<double>> v2) {
    if (v1[0].size() != v2.size()) {
        cout << "请输入维度正确的矩阵！" << endl;
    }

    int new_d1 = v1.size();
    int new_d2 = v2[0].size();
    vector<vector<double>> v(new_d1, vector<double>(new_d2));

    // 点乘运算
    for (int i=0;i<new_d1;i++) {
        for (int j=0;j<new_d2;j++) {
            double cell = 0;
            for (int c=0;c<v1[0].size();c++) {
                cell += v1[i][c] * v2[c][j];
            }
            v[i][j] = cell;
        }
    }
    return v;
}

// 计算loss
double mse_loss(vector<vector<double>> v1, vector<vector<double>> v2) {
    // ensure the shape of v1 equals with the shape of v2
    if (v1.size() != v2.size() | v1[0].size() != v2[0].size()) {
        cout << "维度不一致" << endl;
        return (double)0;
    }

    double res = 0;
    for (int i=0;i<v1.size();i++) {
        for (int j=0;j<v1[0].size();j++) {
            res += pow((v1[i][j] - v2[i][j]), 2);
        }
    }
    return res;
}


// 乘常数
vector<vector<double>> const_dot(vector<vector<double>> v1, double c) {
    for (int i=0;i<v1.size();i++) {
        for (int j=0;j<v1[0].size();j++) {
            v1[i][j] = v1[i][j] * c;
        }
    }
    return v1;
}


// 矩阵相减
vector<vector<double>> vec_minus(vector<vector<double>> v1, vector<vector<double>> v2) {
    vector<vector<double>> v(v1.size(), vector<double>(v1[0].size()));
    for (int i=0;i<v1.size();i++) {
        for (int j=0;j<v1[0].size();j++) {
            v[i][j] = v1[i][j] - v2[i][j];
        }
    }
    return v;
}

//矩阵转置
vector<vector<double>> vec_transpose(vector<vector<double>> v) {
    vector<vector<double>> v1(v[0].size(), vector<double>(v.size()));
    for (int i=0;i<v1.size();i++) {
        for (int j=0;j<v1[0].size();j++) {
            v1[i][j] = v[j][i];
        }
    }
    return v1;
}
