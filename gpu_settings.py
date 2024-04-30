def gpu_settings1():

    print()
    print("GPU Settings1: returns device_name in GPU_Settings.py")

    import tensorflow as tf

    # Get the GPU device name,
    device_name = tf.test.gpu_device_name()

    # The device name should look like the follwing:
    if device_name == '/device:GPU:0':
      print('Found GPU at: {}'.format(device_name))
    else:
      raise SystemError('GPU device not found')

    return device_name

def gpu_settings2():

    print()
    print("GPU Settings2: returns device in GPU_Settings.py")

    import torch

    # It there's a GPU available...
    if torch.cuda.is_available():

      # Tell PyTorch to use the GPU.
      device = torch.device("cuda")

      print('There are %d GPU(s) available.' % torch.cuda.device_count())

      print('We will use the GPU:' , torch.cuda.get_device_name(0))

    # If not
    else:
      print('No GPU is available, using CPU instead.')
      device = torch.device('cpu')

    return device



def gpu_settings3():

    import torch

    print()
    print("GPU Settings3: returns device in GPU_Settings.py")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device
    return device