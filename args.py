class Args():
    # 训练参数
    path_ir = ''
    cuda = 1
    lr = 1e-4
    epochs = 20
    batch_size = 4
    device = 0;

    # 网络参数
    Height = 128
    Width = 128

    n = 64  # number of filters
    channel = 1  # 1 - gray, 3 - RGB
    s = 3  # filter size
    stride = 1
    num_block = 4  
    train_num = 10000

    resume_model = None
    save_fusion_model = "./model"
    save_loss_dir = "./model/loss_v1"
