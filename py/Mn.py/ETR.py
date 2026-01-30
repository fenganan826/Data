import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
import shap

# 初始化SHAP
shap.initjs()

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "SimSun", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 读取数据
file_path = r'F:/Mn_standardized.xlsx'
df = pd.read_excel(file_path)

# 特征与目标
X = df.drop('Mn(×104g/mol）', axis=1)
y = df['Mn(×104g/mol）']

# 区分类别型和数值型特征
categorical_cols = ['M_Zr', 'M_Hf', 'M_Ti', 'R3']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# 数据划分为 80% 训练集 和 20% 测试集
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建预处理流水线
preprocessor = ColumnTransformer(
    transformers=[('num', 'passthrough', numeric_cols),
                  ('cat', OneHotEncoder(), categorical_cols)])

# 定义目标函数，用于贝叶斯优化，目标是最小化MSE，并使用10折交叉验证
def objective(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth) if max_depth is not None else None
    min_samples_split = int(min_samples_split)
    min_samples_leaf = int(min_samples_leaf)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', ExtraTreesRegressor(n_estimators=n_estimators,
                                                                 max_depth=max_depth,
                                                                 min_samples_split=min_samples_split,
                                                                 min_samples_leaf=min_samples_leaf,
                                                                 random_state=42))])

    # 使用10折交叉验证评估模型
    scores = cross_val_score(pipeline, X_trainval, y_trainval,
                             scoring='neg_mean_squared_error',
                             cv=KFold(n_splits=10, shuffle=True, random_state=42))
    return scores.mean()  # 贝叶斯优化最大化目标函数

# 定义参数搜索范围，移除了max_features
pbounds = {
    'n_estimators': (10, 100),  # 更多树，提升稳定性
    'max_depth': (3, 10),  # 限制深度
    'min_samples_split': (2, 15),
    'min_samples_leaf': (1, 5)
}

# 初始化贝叶斯优化器
optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42,
)

# 进行优化，设置迭代次数
optimizer.maximize(
    init_points=5,
    n_iter=10,
)

# 获取最佳参数
best_params = optimizer.max['params']
best_n_estimators = int(best_params['n_estimators'])
best_max_depth = int(best_params['max_depth']) if 'max_depth' in best_params else None
best_min_samples_split = int(best_params['min_samples_split'])
best_min_samples_leaf = int(best_params['min_samples_leaf'])

# 使用最佳参数构建最终模型
final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', ExtraTreesRegressor(n_estimators=best_n_estimators,
                                                                   max_depth=best_max_depth,
                                                                   min_samples_split=best_min_samples_split,
                                                                   min_samples_leaf=best_min_samples_leaf,
                                                                   random_state=42))])

# 使用整个训练集重新训练模型
final_pipeline.fit(X_trainval, y_trainval)

# 在测试集上进行预测
y_pred_test = final_pipeline.predict(X_test)

# 测试集评估
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("\n测试集评估结果：")
print(f"RMSE: {rmse_test}")
print(f"MAE: {mae_test}")
print(f"R²: {r2_test}")

print(f"\n最佳参数: n_estimators={best_n_estimators}, max_depth={best_max_depth}, "
      f"min_samples_split={best_min_samples_split}, min_samples_leaf={best_min_samples_leaf}")

# 使用KFold进行交叉验证的训练集和验证集划分
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 用于存储每一折的训练和验证误差（在每个 n_estimators 下）
train_mse_folds = []
val_mse_folds = []
train_r2_folds = []
val_r2_folds = []

# 用于存储每个训练次数下的均值
mean_train_mse = []
mean_val_mse = []
mean_train_r2 = []
mean_val_r2 = []

# 遍历不同的 n_estimators 值进行训练
for n_estimators in np.arange(32, 201, 10):  # 这里的步长可以调整
    fold_train_mse = []
    fold_val_mse = []
    fold_train_r2 = []
    fold_val_r2 = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_trainval), start=1):
        X_train_fold, X_val_fold = X_trainval.iloc[train_index], X_trainval.iloc[val_index]
        y_train_fold, y_val_fold = y_trainval.iloc[train_index], y_trainval.iloc[val_index]

        model = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', ExtraTreesRegressor(
                                   n_estimators=n_estimators,
                                   max_depth=best_max_depth,
                                   min_samples_split=best_min_samples_split,
                                   min_samples_leaf=best_min_samples_leaf,
                                   random_state=42))])

        model.fit(X_train_fold, y_train_fold)

        # 预测训练集和验证集
        y_train_pred = model.predict(X_train_fold)
        y_val_pred = model.predict(X_val_fold)

        # 计算每一折的MSE和R²
        fold_train_mse.append(mean_squared_error(y_train_fold, y_train_pred))
        fold_val_mse.append(mean_squared_error(y_val_fold, y_val_pred))
        fold_train_r2.append(r2_score(y_train_fold, y_train_pred))
        fold_val_r2.append(r2_score(y_val_fold, y_val_pred))

        # 输出每折的训练和验证结果
        print(f"Fold {fold} 训练集 MSE: {fold_train_mse[-1]}, 验证集 MSE: {fold_val_mse[-1]}")
        print(f"Fold {fold} 训练集 R²: {fold_train_r2[-1]}, 验证集 R²: {fold_val_r2[-1]}")

    # 计算每个训练次数下的平均误差和R²
    mean_train_mse.append(np.mean(fold_train_mse))
    mean_val_mse.append(np.mean(fold_val_mse))
    mean_train_r2.append(np.mean(fold_train_r2))
    mean_val_r2.append(np.mean(fold_val_r2))

# 绘制学习曲线
fig, ax1 = plt.subplots(figsize=(10, 6))

# 左轴：每个 n_estimators 的均值 MSE
ax1.plot(np.arange(32, 201, 10), mean_train_mse, 'o-', color='blue', label='训练集 MSE')
ax1.plot(np.arange(32, 201, 10), mean_val_mse, 's-', color='cyan', label='验证集 MSE')
ax1.set_xlabel('训练次数 (n_estimators)')
ax1.set_ylabel('均方误差 (MSE)', color='blue')

# 右轴：均值 R²
ax2 = ax1.twinx()
ax2.plot(np.arange(32, 201, 10), mean_train_r2, 'o--', color='red', label='训练集 R²')
ax2.plot(np.arange(32, 201, 10), mean_val_r2, 's--', color='orange', label='验证集 R²')
ax2.set_ylabel('决定系数 R²', color='red')

# 合并图例
lines = ax1.get_lines() + ax2.get_lines()
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left')

plt.grid(True)
plt.tight_layout()
plt.title('学习曲线：训练集 vs 验证集')
plt.show()
