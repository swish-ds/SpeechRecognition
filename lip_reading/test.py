lrs = [1e-5, 4e-5, 7e-5, 1e-6, 4e-6, 7e-6]
moms = [0.95, 0.90, 0.85, 0.80]

def format_e(n):
        a = '%E' % n
        return a.split('E')[0].rstrip('0').rstrip('.') + 'e-' + a.split('E')[1][-1]

for lr in lrs:
    for mom in moms:
        print(format_e(lr), mom)

