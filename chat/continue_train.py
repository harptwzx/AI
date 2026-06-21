import tensorflow as tf

# 加载模型
model = tf.saved_model.load("saved_model/gpt_language_model")

# 加载新数据
# ... 准备新数据 ...

# 继续训练（需要重新构建训练循环）
# ...
