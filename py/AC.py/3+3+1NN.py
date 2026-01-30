import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.layers import LeakyReLU
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization,
                                     ReLU, ELU, Activation, Concatenate, Flatten)
from tensorflow.keras.optimizers import Adam, AdamW, RMSprop, SGD, Nadam, Adamax, Ftrl
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import regularizers
from bayes_opt import BayesianOptimization
from tensorflow.keras import backend as K

plt.rcParams["font.family"] = ["SimHei", "SimSun", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 数据读取
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

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
set_seed(42)

def r2_metric(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())

activation_map = {
    'relu': ReLU(),
    'elu': ELU(),
    'swish': Activation(tf.nn.swish),
    'gelu': Activation(tf.nn.gelu),
    'leaky_relu': LeakyReLU()
}
activation_choices = list(activation_map.keys())

pbounds = {
    'units_num': (64, 256),
    'units_cat': (64, 140),
    'dropout_num_1': (0.1, 0.4),
    'dropout_num_2': (0.1, 0.4),
    'dropout_num_3': (0.1, 0.5),
    'dropout_cat_1': (0.1, 0.4),
    'dropout_cat_2': (0.2, 0.7),
    'dropout_cat_3': (0.1, 0.2),
    'learning_rate': (3e-4, 8e-4),
    'l2_reg': (5e-6, 5e-5),
    'batch_size': (40, 64),
    'fusion_dim': (128, 400),
    'act_choice_idx': (0, len(activation_choices) - 1),
    'optimizer_choice_idx': (0, 5)
}

def build_and_evaluate_model(units_num, units_cat,
                             dropout_num_1, dropout_num_2, dropout_num_3,
                             dropout_cat_1, dropout_cat_2, dropout_cat_3,
                             learning_rate, l2_reg, batch_size,
                             act_choice_idx, fusion_dim, optimizer_choice_idx):
    units_num = int(units_num)
    units_cat = int(units_cat)
    batch_size = int(batch_size)
    fusion_dim = int(fusion_dim)
    act_choice = activation_choices[int(act_choice_idx)]
    act_layer = activation_map[act_choice]

    optimizer = [
        AdamW(learning_rate=learning_rate, weight_decay=l2_reg),
        RMSprop(learning_rate=learning_rate),
        SGD(learning_rate=learning_rate),
        Nadam(learning_rate=learning_rate),
        Adamax(learning_rate=learning_rate),
        Ftrl(learning_rate=learning_rate)
    ][int(optimizer_choice_idx)]

    input_num = Input(shape=(X_num_train.shape[1],))
    x_num = Dense(units_num, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=l2_reg))(input_num)
    x_num = BatchNormalization()(x_num)
    x_num = act_layer(x_num)
    x_num = Dropout(dropout_num_1)(x_num)
    x_num = Dense(units_num // 2)(x_num)
    x_num = act_layer(x_num)
    x_num = Dropout(dropout_num_2)(x_num)
    x_num = Dense(units_num // 4)(x_num)
    x_num = act_layer(x_num)
    x_num = Dropout(dropout_num_3)(x_num)
    x_num = Flatten()(x_num)

    input_cat = Input(shape=(X_cat_train.shape[1],))
    x_cat = Dense(units_cat)(input_cat)
    x_cat = act_layer(x_cat)
    x_cat = Dropout(dropout_cat_1)(x_cat)
    x_cat = Dense(units_cat // 2)(x_cat)
    x_cat = act_layer(x_cat)
    x_cat = Dropout(dropout_cat_2)(x_cat)
    x_cat = Dense(units_cat // 4)(x_cat)
    x_cat = act_layer(x_cat)
    x_cat = Dropout(dropout_cat_3)(x_cat)
    x_cat = Flatten()(x_cat)

    x = Concatenate()([x_num, x_cat])
    x = Dense(fusion_dim, activation='relu')(x)
    output = Dense(1)(x)

    model = Model(inputs=[input_num, input_cat], outputs=output)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.Huber(delta=1.0),
                  metrics=['mae', r2_metric])

    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

    model.fit([X_num_train, X_cat_train], y_train,
              validation_data=([X_num_val, X_cat_val], y_val),
              epochs=300,
              batch_size=batch_size,
              verbose=0,
              callbacks=[early_stop, reduce_lr])

    y_val_pred = model.predict([X_num_val, X_cat_val]).flatten()
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    score = 0.3 * val_mse + 0.7 * (1 - val_r2)
    return -score

# 开始贝叶斯优化
optimizer = BayesianOptimization(
    f=build_and_evaluate_model,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)
optimizer.maximize(init_points=10, n_iter=50)

# 最优参数结果整理
best = optimizer.max['params']
for key in ['units_num', 'units_cat', 'batch_size', 'fusion_dim', 'act_choice_idx', 'optimizer_choice_idx']:
    best[key] = int(best[key])
best['act_choice'] = activation_choices[best['act_choice_idx']]
act_layer = activation_map[best['act_choice']]

print("\n📌 贝叶斯优化得到的最佳超参数：")
for k, v in best.items():
    print(f"{k}: {v}")

# 构建最终模型
input_num = Input(shape=(X_num.shape[1],))
x_num = Dense(best['units_num'], kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=best['l2_reg']))(input_num)
x_num = BatchNormalization()(x_num)
x_num = act_layer(x_num)
x_num = Dropout(best['dropout_num_1'])(x_num)
x_num = Dense(best['units_num'] // 2)(x_num)
x_num = act_layer(x_num)
x_num = Dropout(best['dropout_num_2'])(x_num)
x_num = Dense(best['units_num'] // 4)(x_num)
x_num = act_layer(x_num)
x_num = Dropout(best['dropout_num_3'])(x_num)
x_num = Flatten()(x_num)

input_cat = Input(shape=(X_cat.shape[1],))
x_cat = Dense(best['units_cat'])(input_cat)
x_cat = act_layer(x_cat)
x_cat = Dropout(best['dropout_cat_1'])(x_cat)
x_cat = Dense(best['units_cat'] // 2)(x_cat)
x_cat = act_layer(x_cat)
x_cat = Dropout(best['dropout_cat_2'])(x_cat)
x_cat = Dense(best['units_cat'] // 4)(x_cat)
x_cat = act_layer(x_cat)
x_cat = Dropout(best['dropout_cat_3'])(x_cat)
x_cat = Flatten()(x_cat)

x = Concatenate()([x_num, x_cat])
x = Dense(best['fusion_dim'], activation='relu')(x)
output = Dense(1)(x)

model = Model(inputs=[input_num, input_cat], outputs=output)
optimizer_list = [AdamW(learning_rate=best['learning_rate'], weight_decay=best['l2_reg']),
                  RMSprop(learning_rate=best['learning_rate']),
                  SGD(learning_rate=best['learning_rate']),
                  Nadam(learning_rate=best['learning_rate']),
                  Adamax(learning_rate=best['learning_rate']),
                  Ftrl(learning_rate=best['learning_rate'])]

best_optimizer = optimizer_list[best['optimizer_choice_idx']]
model.compile(optimizer=best_optimizer,
              loss=tf.keras.losses.Huber(delta=1.0),
              metrics=['mae', r2_metric])

checkpoint_path = "best_model_weights.h5"
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                   monitor='val_loss',
                                   save_best_only=True,
                                   save_weights_only=True,
                                   verbose=1)

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

history = model.fit([X_num_train, X_cat_train], y_train,
                    validation_data=([X_num_val, X_cat_val], y_val),
                    epochs=300,
                    batch_size=best['batch_size'],
                    verbose=1,
                    callbacks=[early_stop, reduce_lr, model_checkpoint])

model.load_weights(checkpoint_path)

# 评估与可视化
y_pred_train = model.predict([X_num_train, X_cat_train]).flatten()
y_pred_val = model.predict([X_num_val, X_cat_val]).flatten()
y_pred_test = model.predict([X_num_test, X_cat_test]).flatten()

# 将训练集、验证集和测试集的真实值与预测值保存到 CSV 文件
df_all = pd.DataFrame({
    '真实值': np.concatenate([y_train, y_val, y_test]),
    '预测值': np.concatenate([y_pred_train, y_pred_val, y_pred_test])
})

# 保存到 F 盘
df_all.to_csv(r'F:/all_data_predictions.csv', index=False, encoding='utf-8-sig')
print("✅ 所有数据集（训练集、验证集、测试集）的真实值与预测值已保存至 F:/all_data_predictions.csv")
