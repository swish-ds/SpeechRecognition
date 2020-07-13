import tensorflow
from tensorflow.keras.utils import plot_model
from models.lr_models import LipNetNorm2, LipNetNorm6
from utils import global_params

model = LipNetNorm2()
model.model().summary()
plot_model(model.model(), show_shapes=True, to_file='LipNetNorm2.png', expand_nested=True)

