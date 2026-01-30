import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization,
                                     ReLU, ELU, Concatenate, Activation, Flatten)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

from bayes_opt import BayesianOptimization

# ====================== 全局配置 ======================
plt.rcParams["font.family"] = ["SimHei", "SimSun", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 固定随机种子
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

# 数据读取与预处理
file_path = r'F:/PDI_standardized.xlsx'
df = pd.read_excel(file_path)

numerical_cols = ['Cat(umol)', 'Al/M(molar)', 't/min', 'T/℃', 'R1', 'R2']
one_hot_cols = ['M_Zr', 'M_Ti', 'M_Hf', 'R3']
target_col = 'PDI'

X_num = df[numerical_cols].values
X_cat = df[one_hot_cols].values
y = df[target_col].values

# 划分训练集/测试集（80/20）
X_num_trainval, X_num_test, X_cat_trainval, X_cat_test, y_trainval, y_test = train_test_split(
    X_num, X_cat, y, test_size=0.2, random_state=42
)

# ====================== 自定义 R² 指标 ======================
def r2_metric(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())

# ====================== 激活函数映射 ======================
activation_map = {
    'relu': ReLU(),
    'elu': ELU(),
    'swish': Activation(tf.nn.swish),
    'gelu': Activation(tf.nn.gelu)
}
activation_choices = list(activation_map.keys())

# ====================== 贝叶斯优化目标函数（10折交叉验证，打印每折结果） ======================
def build_and_evaluate_model_cv(units_num, units_cat,
                                dropout_num1, dropout_num2, dropout_num3,
                                dropout_cat1, dropout_cat2, dropout_cat3,
                                learning_rate, l2_reg, batch_size,
                                act_choice_idx, fusion_dim):

    units_num = int(units_num)
    units_cat = int(units_cat)
    batch_size = int(batch_size)
    fusion_dim = int(fusion_dim)
    act_choice = activation_choices[int(act_choice_idx)]
    act_layer = activation_map[act_choice]

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    val_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_num_trainval), start=1):
        print(f"第 {fold} 折交叉验证...")
        Xn_train, Xn_val = X_num_trainval[train_idx], X_num_trainval[val_idx]
        Xc_train, Xc_val = X_cat_trainval[train_idx], X_cat_trainval[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

        # 数值分支
        input_num = Input(shape=(Xn_train.shape[1],))
        x_num = Dense(units_num, kernel_regularizer=regularizers.l1_l2(1e-5, l2_reg))(input_num)
        x_num = BatchNormalization()(x_num)
        x_num = act_layer(x_num)
        x_num = Dropout(dropout_num1)(x_num)
        x_num = Dense(units_num // 2)(x_num)
        x_num = act_layer(x_num)
        x_num = Dropout(dropout_num2)(x_num)
        x_num = Dense(units_num // 4)(x_num)
        x_num = act_layer(x_num)
        x_num = Dropout(dropout_num3)(x_num)
        x_num = Flatten()(x_num)

        # 分类分支
        input_cat = Input(shape=(Xc_train.shape[1],))
        x_cat = Dense(units_cat)(input_cat)
        x_cat = act_layer(x_cat)
        x_cat = Dropout(dropout_cat1)(x_cat)
        x_cat = Dense(units_cat // 2)(x_cat)
        x_cat = act_layer(x_cat)
        x_cat = Dropout(dropout_cat2)(x_cat)
        x_cat = Dense(units_cat // 4)(x_cat)
        x_cat = act_layer(x_cat)
        x_cat = Dropout(dropout_cat3)(x_cat)
        x_cat = Flatten()(x_cat)

        # 融合 + 输出
        x = Concatenate()([x_num, x_cat])
        x = Dense(fusion_dim, activation='relu')(x)
        output = Dense(1)(x)

        model = Model(inputs=[input_num, input_cat], outputs=output)

        optimizer = AdamW(learning_rate=learning_rate, weight_decay=l2_reg)
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.Huber(delta=1.0),  # BO阶段使用 Huber
                      metrics=['mae', r2_metric])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
        ]

        history = model.fit([Xn_train, Xc_train], y_train,
                            validation_data=([Xn_val, Xc_val], y_val),
                            epochs=200,
                            batch_size=batch_size,
                            verbose=0,
                            callbacks=callbacks)

        # 打印每折最终训练/验证指标（注意：此处 loss 为 Huber，不是 MSE）
        print(f"Fold {fold} 训练集 R²: {history.history['r2_metric'][-1]:.6f}, 验证集 R²: {history.history['val_r2_metric'][-1]:.6f}")
        print(f"Fold {fold} 训练集 Huber: {history.history['loss'][-1]:.6f}, 验证集 Huber: {history.history['val_loss'][-1]:.6f}")

        # 验证集评估用于综合分数（sklearn 计算 MSE/R²）
        y_val_pred = model.predict([Xn_val, Xc_val]).flatten()
        mse = mean_squared_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)
        score = 0.3 * mse + 0.7 * (1 - r2)  # 综合指标
        val_scores.append(score)

    return -np.mean(val_scores)

# ====================== 贝叶斯优化参数空间 ======================
pbounds = {
    'units_num': (64, 128),
    'units_cat': (32, 100),
    'fusion_dim': (64, 128),
    'dropout_num1': (0.1, 0.3),
    'dropout_num2': (0.1, 0.3),
    'dropout_num3': (0.1, 0.3),
    'dropout_cat1': (0.1, 0.3),
    'dropout_cat2': (0.1, 0.3),
    'dropout_cat3': (0.1, 0.3),
    'learning_rate': (3e-4, 8e-4),
    'l2_reg': (5e-6, 5e-5),
    'batch_size': (32, 64),
    'act_choice_idx': (0, len(activation_choices) - 1)
}

optimizer = BayesianOptimization(
    f=build_and_evaluate_model_cv,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points=5, n_iter=10)

best = optimizer.max['params']
for key in ['units_num', 'units_cat', 'fusion_dim', 'batch_size', 'act_choice_idx']:
    best[key] = int(best[key])
best['act_choice'] = activation_choices[best['act_choice_idx']]

print("\n📌 最佳超参数：")
for k, v in best.items():
    print(f"{k}: {v}")

# ====================== 构建交叉验证学习曲线（按 epoch 取10折均值，num_epochs=200；用 MSE） ======================
kf = KFold(n_splits=10, shuffle=True, random_state=42)
num_epochs = 200  # 统一为200

train_mse_all = np.zeros((num_epochs, 10))
val_mse_all = np.zeros((num_epochs, 10))
train_r2_all = np.zeros((num_epochs, 10))
val_r2_all = np.zeros((num_epochs, 10))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_num_trainval), start=1):
    Xn_train, Xn_val = X_num_trainval[train_idx], X_num_trainval[val_idx]
    Xc_train, Xc_val = X_cat_trainval[train_idx], X_cat_trainval[val_idx]
    y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

    act_layer = activation_map[best['act_choice']]

    input_num = Input(shape=(Xn_train.shape[1],))
    x_num = Dense(best['units_num'], kernel_regularizer=regularizers.l1_l2(1e-5, best['l2_reg']))(input_num)
    x_num = BatchNormalization()(x_num)
    x_num = act_layer(x_num)
    x_num = Dropout(best['dropout_num1'])(x_num)
    x_num = Dense(best['units_num'] // 2)(x_num)
    x_num = act_layer(x_num)
    x_num = Dropout(best['dropout_num2'])(x_num)
    x_num = Dense(best['units_num'] // 4)(x_num)
    x_num = act_layer(x_num)
    x_num = Dropout(best['dropout_num3'])(x_num)
    x_num = Flatten()(x_num)

    input_cat = Input(shape=(Xc_train.shape[1],))
    x_cat = Dense(best['units_cat'])(input_cat)
    x_cat = act_layer(x_cat)
    x_cat = Dropout(best['dropout_cat1'])(x_cat)
    x_cat = Dense(best['units_cat'] // 2)(x_cat)
    x_cat = act_layer(x_cat)
    x_cat = Dropout(best['dropout_cat2'])(x_cat)
    x_cat = Dense(best['units_cat'] // 4)(x_cat)
    x_cat = act_layer(x_cat)
    x_cat = Dropout(best['dropout_cat3'])(x_cat)
    x_cat = Flatten()(x_cat)

    x = Concatenate()([x_num, x_cat])
    x = Dense(best['fusion_dim'], activation='relu')(x)
    output = Dense(1)(x)

    model = Model(inputs=[input_num, input_cat], outputs=output)

    optimizer_final = AdamW(learning_rate=best['learning_rate'], weight_decay=best['l2_reg'])
    # 这里显式使用 MSE 作为损失，以便“学习曲线=MSE”的语义完全一致
    model.compile(optimizer=optimizer_final, loss='mse', metrics=[r2_metric])

    history = model.fit([Xn_train, Xc_train], y_train,
                        validation_data=([Xn_val, Xc_val], y_val),
                        epochs=num_epochs,
                        batch_size=best['batch_size'],
                        verbose=0)

    train_mse_all[:, fold-1] = history.history['loss']       # MSE
    val_mse_all[:, fold-1]  = history.history['val_loss']    # MSE
    train_r2_all[:, fold-1] = history.history['r2_metric']   # R²
    val_r2_all[:, fold-1]   = history.history['val_r2_metric']

# ====================== 学习曲线绘图（10 折均值；MSE + R²） ======================
mean_train_mse = train_mse_all.mean(axis=1)
mean_val_mse = val_mse_all.mean(axis=1)
mean_train_r2 = train_r2_all.mean(axis=1)
mean_val_r2 = val_r2_all.mean(axis=1)

epochs_range = range(1, num_epochs + 1)

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_xlabel('训练轮数 (Epoch)')
ax1.set_ylabel('均方误差 (MSE)', color='tab:blue')
l1 = ax1.plot(epochs_range, mean_train_mse, label='训练集 MSE', color='tab:blue')
l2 = ax1.plot(epochs_range, mean_val_mse, label='验证集 MSE', color='tab:cyan')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('决定系数 R²', color='tab:red')
l3 = ax2.plot(epochs_range, mean_train_r2, '--', label='训练集 R²', color='tab:red')
l4 = ax2.plot(epochs_range, mean_val_r2, '--', label='验证集 R²', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylim(0, 1.05)
ax2.set_yticks(np.arange(0, 1.1, 0.1))

lines = l1 + l2 + l3 + l4
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='lower center', ncol=4)
plt.title('神经网络学习曲线（10折交叉验证均值，Epoch=200；损失=MSE）')
plt.grid(True)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()

# ====================== 最终模型与测试评估（不再留出验证集，直接用整个训练集训练） ======================
act_layer = activation_map[best['act_choice']]

input_num = Input(shape=(X_num.shape[1],))
x_num = Dense(best['units_num'], kernel_regularizer=regularizers.l1_l2(1e-5, best['l2_reg']))(input_num)
x_num = BatchNormalization()(x_num)
x_num = act_layer(x_num)
x_num = Dropout(best['dropout_num1'])(x_num)
x_num = Dense(best['units_num'] // 2)(x_num)
x_num = act_layer(x_num)
x_num = Dropout(best['dropout_num2'])(x_num)
x_num = Dense(best['units_num'] // 4)(x_num)
x_num = act_layer(x_num)
x_num = Dropout(best['dropout_num3'])(x_num)
x_num = Flatten()(x_num)

input_cat = Input(shape=(X_cat.shape[1],))
x_cat = Dense(best['units_cat'])(input_cat)
x_cat = act_layer(x_cat)
x_cat = Dropout(best['dropout_cat1'])(x_cat)
x_cat = Dense(best['units_cat'] // 2)(x_cat)
x_cat = act_layer(x_cat)
x_cat = Dropout(best['dropout_cat2'])(x_cat)
x_cat = Dense(best['units_cat'] // 4)(x_cat)
x_cat = act_layer(x_cat)
x_cat = Dropout(best['dropout_cat3'])(x_cat)
x_cat = Flatten()(x_cat)

x = Concatenate()([x_num, x_cat])
x = Dense(best['fusion_dim'], activation='relu')(x)
output = Dense(1)(x)

final_model = Model(inputs=[input_num, input_cat], outputs=output)

optimizer_final = AdamW(learning_rate=best['learning_rate'], weight_decay=best['l2_reg'])
# 与 BO 阶段保持一致：使用 Huber（更抗异常值）；若希望与学习曲线完全一致，也可改为 'mse'
final_model.compile(optimizer=optimizer_final,
                    loss=tf.keras.losses.Huber(delta=1.0),
                    metrics=['mae', r2_metric])

# 不再留出验证集，直接在整个训练集上训练；回调监控训练损失
callbacks_final = [
    EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='loss', factor=0.5, patience=8, min_lr=1e-6)
]

history_final = final_model.fit([X_num_trainval, X_cat_trainval], y_trainval,
                                epochs=200,
                                batch_size=best['batch_size'],
                                verbose=1,
                                callbacks=callbacks_final)

# ====================== 测试集评估 ======================
y_pred_test = final_model.predict([X_num_test, X_cat_test]).flatten()

mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("\n📊 最终模型评估结果（测试集）：")
print(f"测试集 MSE: {mse_test}")
print(f"测试集 MAE: {mae_test}")
print(f"测试集 R²: {r2_test}")

# ====================== 真实值 vs 预测值 散点图 ======================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, c='dodgerblue', alpha=0.7, edgecolors='k', label='预测点')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='理想预测线')
plt.xlabel('真实值', fontsize=12)
plt.ylabel('预测值', fontsize=12)
plt.title('测试集：真实值 vs 预测值', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ====================== 保存预测结果 ======================
test_result = pd.DataFrame({'真实值': y_test, '预测值': y_pred_test})
test_result.to_excel(r'F:/test_pred_vs_true.xlsx', index=False)
print("✅ 已将测试集真实值与预测值保存到 F:/test_pred_vs_true.xlsx")
