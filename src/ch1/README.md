## 安装

本章的代码需要安装Python，Numpy和Matplotlib。还需要安装Jupyter Notebook。建议用virtualenv里创建一个隔离的环境。本书大部分代码假设使用Python3.6，其它版本的Python也可能可以工作，但是没有经过严格的测试。

virtualenv的安装请参考[官方文档](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)。在virtualenv里安装Jupter Notebook请参考[这里](https://medium.com/@eleroy/jupyter-notebook-in-a-virtual-environment-virtualenv-8f3c3448247)。

## 获取代码
注：只需要运行一次。
```
git clone https://github.com/fancyerii/deep_learning_theory_and_practice.git
cd deep_learning_theory_and_practice
```

可以定期使用"git pull"拉取更新。

## 运行

```
$ cd /src/ch1
$ jupyter notebook
```

浏览器会自动打开类似于"[http://localhost:8888/tree](http://localhost:8888/tree)"的地址，如果没有自动打开，去上命令的输出中寻找url，用浏览器打开，然后寻找"Linear Regression.ipynb"。

Jupyter Notebook的用法可以参考[官方文档](https://jupyter-notebook.readthedocs.io/en/stable/)或者[Jupyter Notebook: An Introduction](https://realpython.com/jupyter-notebook-introduction/)。
