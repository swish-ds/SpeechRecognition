import tensorflow as tf
from tensorflow import keras
from models.lr_models import LipNetNorm
from utils import global_params

ln = LipNetNorm(batch_s=global_params.batch_s, frames_n=global_params.frames_n, img_h=global_params.img_h, img_w=global_params.img_w,
                img_c=global_params.img_c, dropout_s=global_params.dropout_s, output_size=global_params.classes_n)

tf.keras.utils.plot_model(ln, "model_arch.png")

loss_func = keras.losses.CategoricalCrossentropy(
    from_logits=True, label_smoothing=0)
sgd = keras.optimizers.SGD(learning_rate=global_params.lr, momentum=global_params.mom, nesterov=True)

ln.compile(optimizer=sgd, loss=loss_func, metrics=['accuracy'])
ln.model().summary()
