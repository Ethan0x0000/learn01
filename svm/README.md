# SVM 算法实现文档

## 1. 简介
本项目实现了基于支持向量机（Support Vector Machine, SVM）的分类算法。代码基于 Python 的 `scikit-learn` 库，实现了数据预处理、模型训练（包含超参数网格搜索）、模型评估以及决策边界的可视化。

## 2. 算法原理
SVM 是一种监督学习算法，用于分类和回归分析。其核心思想是找到一个超平面（Hyperplane），使得不同类别的样本点到该超平面的间隔（Margin）最大化。
- **最大间隔**：SVM 试图最大化最近的训练数据点（支持向量）与超平面之间的距离。
- **核技巧（Kernel Trick）**：通过将数据映射到高维空间，SVM 可以有效地处理非线性分类问题。常用的核函数包括线性核（Linear）、多项式核（Polynomial）和径向基函数核（RBF）。

## 3. 实现细节
### 3.1 代码结构
核心代码位于 `svm_classifier.py` 文件中，主要包含 `SVMTask` 类，封装了以下功能：
- **数据加载 (`load_data`)**：支持加载 Iris（鸢尾花）和 Digits（手写数字）数据集。
- **预处理 (`preprocess`)**：
  - 数据集划分：训练集/测试集比例为 8:2。
  - 特征标准化：使用 `StandardScaler` 进行 Z-score 标准化。
- **模型训练 (`train`)**：
  - 支持 `GridSearchCV` 自动调优超参数 `C` (惩罚系数) 和 `gamma` (核函数系数)。
  - 支持 Linear 和 RBF 核函数。
- **评估 (`evaluate`)**：输出准确率、分类报告（Precision, Recall, F1-score）和混淆矩阵。
- **可视化 (`visualize`)**：
  - 使用 PCA 将数据降维至 2 维。
  - 绘制决策边界和支持向量。

### 3.2 依赖库
- `numpy`: 数值计算
- `scikit-learn`: 机器学习算法及数据集
- `matplotlib`: 绘图和可视化

## 4. API 说明
### 类 `SVMTask`
```python
class SVMTask(dataset_name='iris', test_size=0.2, random_state=42)
```
- **参数**:
  - `dataset_name`: 数据集名称，可选 `'iris'` 或 `'digits'`。
  - `test_size`: 测试集占比，默认 0.2。
  - `random_state`: 随机种子。

### 主要方法
- `run()`: 执行完整流程（加载->预处理->训练->评估->可视化）。

## 5. 使用示例

### 运行环境准备
确保安装了必要的依赖库：
```bash
pip install numpy scikit-learn matplotlib
```

### 运行代码
可以通过命令行直接运行脚本：

**1. 运行 Iris 数据集任务（默认）**
```bash
python svm_classifier.py --dataset iris
```

**2. 运行 Digits 数据集任务**
```bash
python svm_classifier.py --dataset digits
```

### 输出结果
- 控制台将打印训练过程日志、最佳参数、准确率及详细的评估报告。
- 可视化结果将保存为图片文件：
  - `svm_visualization_iris.png`
  - `svm_visualization_digits.png`

## 6. 数据集说明
本项目使用 `scikit-learn` 内置的标准数据集，无需手动下载。

### 6.1 Iris 数据集
- **来源**: `sklearn.datasets.load_iris()`
- **描述**: 包含 150 个样本，3 个类别（Setosa, Versicolor, Virginica），每个样本 4 个特征。
- **用途**: 适用于测试基础分类效果和可视化决策边界。

### 6.2 Digits 数据集
- **来源**: `sklearn.datasets.load_digits()`
- **描述**: 包含 1797 个样本，10 个类别（数字 0-9），每个样本为 8x8 像素的图像（64 个特征）。
- **用途**: 适用于测试高维数据的分类能力及非线性核函数的效果。

## 7. 注意事项
- **可视化限制**: 由于 PCA 降维会丢失部分信息，二维平面上的决策边界仅作为模型在低维空间投影的直观展示，并不完全代表高维空间中的真实分类边界。
- **训练时间**: 对于大数据集或复杂的参数网格，Grid Search 可能会消耗较多时间。
