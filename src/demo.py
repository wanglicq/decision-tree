import pandas as pd
from sklearn import model_selection
from sklearn import tree
import graphviz

# Step 1, read data
# read csv data
watermelonData = pd.read_csv('../docs/waterMelon.csv')
# read first 5 lines data
#print(watermelonData.head())
# get data dimension
#print(watermelonData.shape)


# Step 2, prepare data
# select sample characteristic
X = watermelonData[['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率']]

# select sample output
y = watermelonData[['好瓜']]

# Step 3, split data into training set and testing set
# cross_validation是交叉验证
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.4)
print(X_train)
print(y_train)
print(X_test.shape)
print(y_test.shape)


# Step 4, train the data, get tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Step 5，模型拟合测试集
y_pred = clf.predict(X_test)



# 可视化决策树
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render('waterMelon', '../docs/watermelon.pdf')

