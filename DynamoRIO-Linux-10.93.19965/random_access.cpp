#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

const size_t ARRAY_SIZE = 1 << 20; // 1 GB
const size_t ACCESS_COUNT = 500;

int main() {
    // 初始化随机数生成器
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // 创建一个大小为 1GB 的数组
    std::vector<int> largeArray(ARRAY_SIZE);

    // 随机访问元素并进行简单操作
    for (size_t i = 0; i < ACCESS_COUNT; ++i) {
        size_t index = std::rand() % ARRAY_SIZE; // 随机生成索引
        largeArray[index] += 1; // 简单操作：增加该位置的值
    }

    // 输出部分结果以确认程序执行
    std::cout << "Completed " << ACCESS_COUNT << " random accesses." << std::endl;
    std::cout << "Sample values: " << largeArray[0] << ", " << largeArray[1] << ", " << largeArray[2] << std::endl;

    return 0;
}