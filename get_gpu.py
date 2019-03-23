def get_gpu():
    import GPUtil
    import os
    import tensorflow as tf
    from tensorflow.python.keras import backend as K
    # Get the first available GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    try:

        DEVICE_ID = GPUtil.getAvailable(order='memory', limit=2, maxLoad=0.001, maxMemory=0.001)[0]

    except:
        print('GPU not compatible with NVIDIA-SMI')
        DEVICE_ID = 'Not Found'
    else:

    # print(DEVICE_ID)
    # DEVICE_ID = (1 + DEVICE_ID) % 2
    # print(DEVICE_ID)
    # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
        # sess = tf.Session(config=tf.ConfigProto())
        config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
       # config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)

    finally:
        # Since all other GPUs are masked out, the first available GPU will now be identified as GPU:0
        # device = '/gpu:0'
        print('Device ID (unmasked): ' + str(DEVICE_ID))
        print('Device ID (masked): ' + str(0))

    return DEVICE_ID
