import pandas as pd
from sklearn import model_selection
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import graphviz
import numpy

#Read CSV file into DataFrame
watermelonData = pd.read_csv('../docs/waterMelon.csv')
dataTransfor = LabelEncoder()

#选择数据特性
watermelonCharactor = watermelonData[['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']].values
print(watermelonData)

# 把字符串都转化为整型
watermelonCharactor[:, 0] = dataTransfor.fit_transform(watermelonCharactor[:, 0])
watermelonCharactor[:, 1] = dataTransfor.fit_transform(watermelonCharactor[:, 1])
watermelonCharactor[:, 2] = dataTransfor.fit_transform(watermelonCharactor[:, 2])
watermelonCharactor[:, 3] = dataTransfor.fit_transform(watermelonCharactor[:, 3])
watermelonCharactor[:, 4] = dataTransfor.fit_transform(watermelonCharactor[:, 4])
watermelonCharactor[:, 5] = dataTransfor.fit_transform(watermelonCharactor[:, 5])

#数据特性进行独热编码
watermelonX = OneHotEncoder(sparse=False).fit_transform(watermelonCharactor)
print(watermelonX.shape)
print(watermelonData[['含糖率']].shape)
watermelonX = numpy.concatenate((watermelonX, watermelonData[['密度']], watermelonData[['含糖率']]), axis=1)
print(watermelonX)
print("watermelonX\n", watermelonX.shape)

#选择样本输出
watermelonOutput = watermelonData[['好瓜']].values
watermelonOutput[:, 0] = dataTransfor.fit_transform(watermelonOutput[:, 0])
watermelonY = OneHotEncoder(sparse=False).fit_transform(watermelonOutput)

# 划分数据为训练集合验证集
X_train, X_test, y_train, y_test = model_selection.train_test_split(watermelonX, watermelonY, test_size = 0.4)

# 训练数据，得到决策树
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

# 拟合验证集
y_pred = clf.predict(X_test)

#计算均方差
print("MSE",metrics.mean_squared_error(y_test, y_pred))


# 可视化决策树
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render('waterMelon_v2', '../docs/watermelon')