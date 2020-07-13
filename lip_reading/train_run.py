from utils import global_params
from scripts import ln_fit

fit = ln_fit.LnFit(model_type=global_params.model_type, optimizer=global_params.optimizer, epochs=global_params.epochs,
                   lr=global_params.lr, mom=global_params.mom, batch_s=global_params.batch_s,
                   classes_n=global_params.classes_n, dropout_s=global_params.dropout_s,
                   frames_n=global_params.frames_n,
                   img_w=global_params.img_w, img_h=global_params.img_h, img_c=global_params.img_c)
# 1e-4
lrs = [1e-4]
moms = [0.90]

for lr in lrs:
    for mom in moms:
        fit.lr = lr
        fit.mom = mom
        fit.train_seq()
