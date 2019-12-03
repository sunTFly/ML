import numpy as np
from timeit import default_timer as timer
from sklearn.pipeline import make_pipeline
from sklearn.datasets.samples_generator import make_blobs, make_circles
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import fetch_lfw_people

'''
make_blobs函数是为聚类产生数据集
产生一个数据集和相应的标签
n_samples:表示数据样本点个数,默认值100
n_features:表示数据的维度，默认值是2
centers:产生数据的中心点，默认值3
cluster_std：数据集的标准差，浮点数或者浮点数序列，默认值1.0
center_box：中心确定之后的数据边界，默认值(-10.0, 10.0)
shuffle ：洗乱，默认值是True
random_state:随机生成器的种子
'''

'''
SVC参数解释
（1）C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0；
（2）kernel：参数选择有RBF, Linear, Poly, Sigmoid, 默认的是"RBF";
（3）degree：if you choose 'Poly' in param 2, this is effective, degree决定了多项式的最高次幂；
（4）gamma：核函数的系数('Poly', 'RBF' and 'Sigmoid'), 默认是gamma = 1 / n_features;
（5）coef0：核函数中的独立项，'RBF' and 'Poly'有效；
（6）probablity: 可能性估计是否使用(true or false)；
（7）shrinking：是否进行启发式；
（8）tol（default = 1e - 3）: svm结束标准的精度;
（9）cache_size: 制定训练所需要的内存（以MB为单位）；
（10）class_weight: 每个类所占据的权重，不同的类设置不同的惩罚参数C, 缺省的话自适应；
（11）verbose: 跟多线程有关，不大明白啥意思具体；
（12）max_iter: 最大迭代次数，default = 1， if max_iter = -1, no limited;
（13）decision_function_shape ： ‘ovo’ 一对一, ‘ovr’ 多对多 or None 无, default=None
（14）random_state ：用于概率估计的数据重排时的伪随机数生成器的种子。
ps：7,8,9一般不考虑。
'''

'''
make_circles 参数环装状数据
'''


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()


def liner():
    X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    model = SVC(kernel='linear', C=0.1)  # 线性模型
    model.fit(X, y)  # 根据数据自适应选择参数
    support = model.support_vectors_  # 得到支持向量
    plot_svc_decision_function(model)


def circle():
    X1, y1 = make_circles(100, factor=.1, noise=.1)
    plt.scatter(X1[:, 0], X1[:, 1], c=y1, s=50, cmap='autumn')
    clf = SVC(kernel='rbf', C=1E6, gamma=0.1).fit(X1, y1)
    support = clf.support_vectors_
    plot_svc_decision_function(clf, plot_support=False)


def facesTest():
    faces = fetch_lfw_people(min_faces_per_person=60)
    print(faces.target_names)
    print(faces.images.shape)
    pca = PCA(n_components=150, whiten=True, random_state=42)
    svc = SVC(kernel='rbf', class_weight='balanced')
    model = make_pipeline(pca, svc)
    Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=40)
    param_grid = {
        'svc__C': [1, 5, 10],
        'svc__gamma': [0.0001, 0.0005, 0.001]
    }
    grid = GridSearchCV(model, param_grid)
    time_start = timer()
    grid.fit(Xtrain, ytrain)
    time_end = timer()
    print('totally cost', time_end - time_start)
    print('------')
    print(grid.best_params_)
    model=grid.best_estimator_
    yfit=model.predict(Xtest)
    print(yfit.shape)
facesTest()
