import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.externals import joblib
from scipy.stats import f


def BuildModel(nAnlysisType):
    if nAnlysisType == 1:
        return LinearRegression()
    elif nAnlysisType == 2:
        return SVR(degree=3, C=1.0)
    elif nAnlysisType == 3:
        return MLPRegressor(hidden_layer_sizes=(10), solver='sgd', momentum=0.9, learning_rate_init=0.001)
    elif nAnlysisType == 4:
        return RandomForestRegressor()
    elif nAnlysisType == 5:
        return GradientBoostingRegressor()
    elif nAnlysisType == 6:
        return BayesianRidge()
    else:
        return LinearRegression()


def BuildSVM(degree, gama):
    return SVR(degree=degree, C=gama)


def BuildMLP(learningRate, momentimRate):
    return MLPRegressor(hidden_layer_sizes=(10), solver='sgd', max_iter=100000, momentum=momentimRate,
                        learning_rate_init=learningRate)


def BuildRandomForest(treeNum, maxDepth, minSplit, minLeaf):
    return RandomForestRegressor(n_estimators=treeNum, max_depth=maxDepth, min_samples_split=minSplit,
                                 min_samples_leaf=minLeaf)


def Train(model, X, Y):
    X = np.asarray(X)
    Y = np.asarray(Y)
    model.fit(X, Y)
    return model


def Test(model, X, Y):
    X = np.asarray(X)
    Y = np.asarray(Y)
    y = model.predict(X)
    ev = explained_variance_score(Y, y)
    mae = mean_absolute_error(Y, y)
    mse = mean_squared_error(Y, y)
    r2 = r2_score(Y, y)
    meany = Y.mean()
    SSe = 0
    SSt = 0
    for i in range(len(y)):
        d = Y[i] - y[i]
        SSe += d * d
        d = Y[i] - meany
        SSt += d * d
    SSr = SSt - SSe
    row, col = X.shape
    DFr = col
    DFe = row - (col + 1)
    DFt = DFr + DFe
    MSe = SSe / DFe
    MSr = SSr / DFr
    MSt = SSt / DFt
    ftest = f.cdf(MSr / MSe, DFr, DFe)
    return [ev, mae, mse, r2, ftest]


def SaveModelExt(model, version, analysisType, nSampleCount, nFactorCount, minDepend, maxDepend, minIndepends,
                 maxIndepends, mae, mse, r2, pvalue, path):
    modelext = {
        'model': model,
        'version': version,
        'analysisType': analysisType,
        'samplecount': nSampleCount,
        'factorcount': nFactorCount,
        'mindepend': minDepend,
        'maxdepend': maxDepend,
        'minIndepends': minIndepends,
        'maxIndepends': maxIndepends,
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'p': pvalue
    }
    joblib.dump(modelext, path)
    return


def LoadModelExt(path):
    modelext = joblib.load(path)
    listmodel = [modelext['model'], modelext['version'], modelext['analysisType'], modelext['samplecount'],
                 modelext['factorcount'], modelext['mindepend'], modelext['maxdepend']]
    for minindepend in modelext['minIndepends']:
        listmodel.append(minindepend)
    for maxindepend in modelext['maxIndepends']:
        listmodel.append(maxindepend)
    listmodel.append(modelext['mae'])
    listmodel.append(modelext['mse'])
    listmodel.append(modelext['r2'])
    listmodel.append(modelext['p'])
    return listmodel


def SaveModel(model, path):
    joblib.dump(model, path)
    return


def LoadModel(path):
    model = joblib.load(path)
    return model


def Predict(model, X):
    X = np.asarray(X)
    Y = model.predict(X)
    return list(Y)


def KFoldTrian(model, X, Y, fold):
    X = np.asarray(X)
    Y = np.asarray(Y)
    kf = KFold(n_splits=fold, shuffle=False)
    mae = float('+inf')
    bestModel = None
    for train_index, test_index in kf.split(X):
        X_Train, X_Test = X[train_index], X[test_index]
        Y_Train, Y_Test = Y[train_index], Y[test_index]
        model = Train(model, X_Train, Y_Train)
        err = Test(model, X_Test, Y_Test)
        if err[1] < mae:
            mae = err[1]
            bestModel = model
    return bestModel


def GetCoefs(model):
    coefs = list()
    for c in model.coef_:
        coefs.append(c)
    coefs.append(model.intercept_)
    return coefs


def main():
    loadedData = datasets.load_boston()
    X = loadedData.data
    Y = loadedData.target
    print(type(X))
    print(X)
    print(type(Y))
    print(Y)

    nFold = 10
    model = BuildModel(1)
    minmaxXScaler = preprocessing.MinMaxScaler()
    XScaler = minmaxXScaler.fit_transform(X)

    minmaxYScaler = preprocessing.MinMaxScaler()

    YScaler = minmaxYScaler.fit_transform(Y.reshape(-1, 1))
    YScaler = YScaler.flatten()

    XTrain, XTest, YTrain, YTest = train_test_split(XScaler, YScaler, test_size=0.3)

    model = KFoldTrian(model, XTrain, YTrain, nFold)

    err = Test(model, XTest, YTest)

    print(err[0])
    print(err[1])
    print(err[2])
    print(err[3])

    SaveModelExt(model, 1, 1, 51, 4, 2.3, 34.2, [1.5, 1.5, 1.5, 1.5], [4, 55, 32.4, 32.7], 0.002, 0.0032, 0.98, 0.99,
                 'D:\\test.model')
    modelext = LoadModelExt('D:\\test.model')

    err = Test(modelext[0], XTest, YTest)

    print(err[0])
    print(err[1])
    print(err[2])
    print(err[3])


if __name__ == '__main__':
    main()
