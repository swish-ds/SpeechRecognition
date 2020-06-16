import ln_fit

fit = ln_fit.LnFit(model_type='norm', optimizer='sgd', epochs=300, img_w=70, img_h=35)

# lrs = [1e-4, 4e-4, 7e-4, 1e-3]
lrs = [1e-3]
moms = [0.90]

for lr in lrs:
    for mom in moms:
        fit.lr = lr
        fit.mom = mom
        fit.train_seq()
