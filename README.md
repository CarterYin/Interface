[//]: # (<br />)
<p align="center">
  <h1 align="center">几个接口的实现</h1>
  <p align="center">
    <img src="https://img.shields.io/badge/Interface-blue?style=flat&logo=github" alt="Interface">
    <img src="https://img.shields.io/badge/Python%20%7C%20Interface-green" alt="Python">
  </p>
</p>

## 要求

这边需要麻烦你封装几个数据集的 eval 代码，方便后续扩展到不同模型和数据集。这里的 RefCOCO 你之前复现过的代码应该是有这部分的，ReasonSeg 我也给了对应的链接。

我在文件里定义了几个接口，主要就是实现这些接口。模型的话可以先自己定义一个容易验证的，或者用你之前复现过的。也可以多找找看其他人有没有类似的实现，然后拿过来改一改。

引入的依赖尽可能少，这样对不同的模型，都可以直接把文件放到他们的项目里，然后 import 他们的模型直接测。