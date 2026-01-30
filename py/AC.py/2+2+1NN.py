import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization,
                                     ReLU, ELU, Concatenate, Activation)
from tensorflow.keras.optimizers import  AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

from bayes_opt import BayesianOptimization

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "SimSun", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 数据读取与预处理
df = pd.read_csv("Activity_standardized.csv")
numerical_cols = ['Cat(umol)', 'Al/M(molar)', 't/min', 'T/C', 'R1', 'R2']
one_hot_cols = ['M_Zr', 'M_Ti', 'M_Hf', 'R3']
target_col = 'Activity(KgPP/mol cat)'

X_num = df[numerical_cols].values
X_cat = df[one_hot_cols].values
y = df[target_col].values

X_num_train, X_num_temp, X_cat_train, X_cat_temp, y_train, y_temp = train_test_split(
    X_num, X_cat, y, test_size=0.3, random_state=42
)
X_num_val, X_num_test, X_cat_val, X_cat_test, y_val, y_test = train_test_split(
    X_num_temp, X_cat_temp, y_temp, test_size=1/3, random_state=42
)

# 固定随机种子
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

# 自定义 R² 评估函数
def r2_metric(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())

# 激活函数映射表
activation_map = {
    'relu': ReLU(),
    'elu': ELU(),
    'swish': Activation(tf.nn.swish),
    'gelu': Activation(tf.nn.gelu)
}
activation_choices = list(activation_map.keys())

# 贝叶斯目标函数
def build_and_evaluate_model(units_num, units_cat, dropout_num, dropout_cat,
                             dropout_num_2, dropout_cat_2,
                             learning_rate, l2_reg, batch_size, act_choice_idx, fusion_dim):
    units_num = int(units_num)
    units_cat = int(units_cat)
    batch_size = int(batch_size)
    fusion_dim = int(fusion_dim)
    act_choice = activation_choices[int(act_choice_idx)]
    act_layer = activation_map[act_choice]

    input_num = Input(shape=(X_num_train.shape[1],))
    x_num = Dense(units_num, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=l2_reg))(input_num)
    x_num = BatchNormalization()(x_num)
    x_num = act_layer(x_num)
    x_num = Dropout(dropout_num)(x_num)
    x_num = Dense(units_num // 2)(x_num)
    x_num = act_layer(x_num)
    x_num = Dropout(dropout_num_2)(x_num)

    input_cat = Input(shape=(X_cat_train.shape[1],))
    x_cat = Dense(units_cat)(input_cat)
    x_cat = act_layer(x_cat)
    x_cat = Dropout(dropout_cat)(x_cat)
    x_cat = Dense(units_cat // 2)(x_cat)
    x_cat = act_layer(x_cat)
    x_cat = Dropout(dropout_cat_2)(x_cat)

    x = Concatenate()([x_num, x_cat])
    x = Dense(fusion_dim, activation='relu')(x)
    output = Dense(1)(x)

    model = Model(inputs=[input_num, input_cat], outputs=output)
    model.compile(optimizer=AdamW(learning_rate=learning_rate, weight_decay=l2_reg),
                  loss=tf.keras.losses.Huber(delta=1.0),
                  metrics=['mae', r2_metric])

    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    model_checkpoint = ModelCheckpoint(
        filepath="best_model_weights.h5",
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=0
    )

    model.fit([X_num_train, X_cat_train], y_train,
              validation_data=([X_num_val, X_cat_val], y_val),
              epochs=300,
              batch_size=batch_size,
              verbose=0,
              callbacks=[early_stop, reduce_lr, model_checkpoint])

    model.load_weights("best_model_weights.h5")

    y_val_pred = model.predict([X_num_val, X_cat_val]).flatten()
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    score = 0.3 * val_mse + 0.7 * (1 - val_r2)
    return -score

pbounds = {
    'units_num': (64, 200),
    'units_cat': (64, 128),
    'fusion_dim': (64, 150),
    'dropout_num': (0.1, 0.2),
    'dropout_cat': (0.1, 0.35),
    'dropout_num_2': (0.1, 0.3),
    'dropout_cat_2': (0.1, 0.2),
    'learning_rate': (5e-5, 5e-4),     # ✅ 降低学习率范围
    'l2_reg': (1e-5, 1e-4),            # ✅ 增强正则
    'batch_size': (64, 100),           # ✅ 稍微增大 batch size
    'act_choice_idx': (0, len(activation_choices) - 1)
}


optimizer = BayesianOptimization(
    f=build_and_evaluate_model,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points=15, n_iter=60)

print("\n\u2705 最佳超参数配置：")
for k, v in optimizer.max['params'].items():
    if k == 'act_choice_idx':
        print(f"{k} = {activation_choices[int(v)]} (index={int(v)})")
    elif isinstance(v, float):
        print(f"{k} = {v}")
    else:
        print(f"{k} = {v}")

# 使用最佳参数重新训练并评估
best = optimizer.max['params']
best['units_num'] = int(best['units_num'])
best['units_cat'] = int(best['units_cat'])
best['batch_size'] = int(best['batch_size'])
best['fusion_dim'] = int(best['fusion_dim'])
best['act_choice_idx'] = int(best['act_choice_idx'])
best['act_choice'] = activation_choices[best['act_choice_idx']]
act_layer = activation_map[best['act_choice']]

input_num = Input(shape=(X_num.shape[1],))
x_num = Dense(best['units_num'], kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=best['l2_reg']))(input_num)
x_num = BatchNormalization()(x_num)
x_num = act_layer(x_num)
x_num = Dropout(best['dropout_num'])(x_num)
x_num = Dense(best['units_num'] // 2)(x_num)
x_num = act_layer(x_num)
x_num = Dropout(0.2)(x_num)


input_cat = Input(shape=(X_cat.shape[1],))
x_cat = Dense(best['units_cat'])(input_cat)
x_cat = act_layer(x_cat)
x_cat = Dropout(best['dropout_cat'])(x_cat)
x_cat = Dense(best['units_cat'] // 2)(x_cat)
x_cat = act_layer(x_cat)
x_cat = Dropout(0.2)(x_cat)


x = Concatenate()([x_num, x_cat])
x = Dense(best['fusion_dim'], activation='relu')(x)
output = Dense(1)(x)

model = Model(inputs=[input_num, input_cat], outputs=output)
model.compile(optimizer=AdamW(learning_rate=best['learning_rate'], weight_decay=best['l2_reg']),
              loss=tf.keras.losses.Huber(delta=1.0),
              metrics=['mae', r2_metric])

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
model_checkpoint = ModelCheckpoint(
    filepath="best_model_weights.h5",
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

history = model.fit([X_num_train, X_cat_train], y_train,
                    validation_data=([X_num_val, X_cat_val], y_val),
                    epochs=300,
                    batch_size=best['batch_size'],
                    verbose=1,
                    callbacks=[early_stop, reduce_lr, model_checkpoint])

# 训练后自动加载最佳权重
model.load_weights("best_model_weights.h5")

# ==========================
# 学习曲线可视化
# ==========================
history_dict = history.history

# 绘制 R² 学习曲线
train_r2 = history_dict['r2_metric']
val_r2 = history_dict['val_r2_metric']

plt.figure(figsize=(12, 6))
plt.plot(train_r2, label='训练集 R²', color='blue')
plt.plot(val_r2, label='验证集 R²', color='orange')
plt.title('训练集和验证集 R² 随训练次数(epoch)的变化')
plt.xlabel('训练次数(epoch)')
plt.ylabel('R²')
plt.legend()
plt.grid(True)
plt.show()

# 绘制 MSE 学习曲线
train_mse = history_dict['loss']
val_mse = history_dict['val_loss']
plt.figure(figsize=(12, 6))
plt.plot(train_mse, label='训练集 MSE', color='blue')
plt.plot(val_mse, label='验证集 MSE', color='orange')
plt.title('训练集和验证集 MSE 随训练次数(epoch)的变化')
plt.xlabel('训练次数(epoch)')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()


# ========================== 双纵坐标轴学习曲线（R² + MSE） ==========================
history_dict = history.history

train_r2 = history_dict['r2_metric']
val_r2 = history_dict['val_r2_metric']
train_mse = history_dict['loss']
val_mse = history_dict['val_loss']
epochs = range(1, len(train_r2) + 1)

fig, ax1 = plt.subplots(figsize=(12, 6))

# 设置统一样式参数
common_kwargs = dict(marker='o', linestyle='--', linewidth=1.5, markersize=5)

# 左轴：MSE
ax1.set_xlabel('训练次数 (epoch)')
ax1.set_ylabel('均方误差 (MSE)', color='tab:blue')
l1 = ax1.plot(epochs, train_mse, color='tab:blue', label='训练 MSE', **common_kwargs)
l2 = ax1.plot(epochs, val_mse, color='tab:cyan', label='验证 MSE', **common_kwargs)
ax1.tick_params(axis='y', labelcolor='tab:blue')

# 右轴：R²
ax2 = ax1.twinx()
ax2.set_ylabel('决定系数 R²', color='tab:red')
l3 = ax2.plot(epochs, train_r2, color='tab:red', label='训练 R²', **common_kwargs)
l4 = ax2.plot(epochs, val_r2, color='tab:orange', label='验证 R²', **common_kwargs)
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylim(0, 1.05)
ax2.set_yticks(np.arange(0, 1.1, 0.1))

# 合并图例
lines = l1 + l2 + l3 + l4
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='lower center', ncol=4)

plt.title('训练过程学习曲线（MSE + R²）')
plt.grid(True)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()


# ========================== 原有的测试集预测导出代码（保持不变） ==========================
y_pred_test = model.predict([X_num_test, X_cat_test]).flatten()

test_result = pd.DataFrame({
    '真实值': y_test,
    '预测值': y_pred_test
})
test_result.to_excel(r'F:/test_pred_vs_true.xlsx', index=False)
print("✅ 已将测试集真实值与预测值保存到 F:/test_pred_vs_true.xlsx")

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

# ========================== 验证集评估（保持不变） ==========================
y_pred_val = model.predict([X_num_val, X_cat_val]).flatten()

mse = mean_squared_error(y_val, y_pred_val)
mae = mean_absolute_error(y_val, y_pred_val)
r2 = r2_score(y_val, y_pred_val)

print("\n📊 最终模型评估结果（验证集）：")
print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")

# ========================== ✅ 新增：测试集评估 ==========================
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("\n📊 最终模型评估结果（测试集）：")
print(f"测试集 MSE: {mse_test}")
print(f"测试集 MAE: {mae_test}")
print(f"测试集 R²: {r2_test}")
