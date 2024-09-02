# ob-vsag
[English](./README.md) | 中文
<div align="left">

[![Chinese Doc](https://img.shields.io/badge/文档-简体中文-blue)]()

</div>

ob-vsag 是 [OceanBase](https://github.com/oceanbase/oceanbase) 团队基于蚂蚁集团[vsag](https://github.com/alipay/vsag)构建的。VSAG 是一个用 C++ 编写的向量索引库，专门用于相似性搜索。该索引算法库提供了基于向量维度和数据规模生成参数的方法，使用户能够搜索不同规模的向量数据集，并且开发人员无需了解算法的具体原理即可进行集成和使用。

## Getting started
gcc9 and cmake-3.22 are required
```
#after clone this project
cd ob-vsag
cmake .
make -j

#after build success,run test example to check
cd example
./hnsw_example
```
- More details refs to [vsag](https://github.com/alipay/vsag)

## Architecture

ob-vsag 与oceanbase整体架构关系如下图所示，oceanbase内核在实现向量检索功能时提供了一组向量检索抽象接口，ob-vsag即在vsag的基础上封装的适配层:

<img src="./images/ob-vsag.jpg" width = "60%" alt="InternalNode" align=center />


# Contributing

热情欢迎每一位对数据库技术热爱的开发者，期待与您携手开启思维碰撞之旅。无论是文档格式调整或文字修正、问题修复还是增加新功能，都是参与和贡献方式之一。

# License

ob-vsag 采用 Apache 2.0 开源协议 
