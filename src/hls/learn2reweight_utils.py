import tensorflow as tf

def updated_theta_copy(grads, variables, lr, mode):
    vals = []
    if mode == 1:
        for g,v in zip(grads,variables):
            vals.append(v+lr*g)
    elif mode == -1:
        for g,v in zip(grads,variables):
            vals.append(v-lr*g)
    else:
        print("invalid mode error!")
        print(exit(1))

    return vals




    