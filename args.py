class Args():
    # 训练参数
    path_ir = ''
    cuda = 1
    lr = 2e-4
    epochs = 20
    batch_size = 8
    device = 0;

    # 网络参数
    Height = 128
    Width = 128

    n = 64  
    channel = 1  
    s = 3  
    stride = 1
    num_block = 4  
    train_num = 5000

    resume_model = None
    save_fusion_model = "./model"
    save_loss_dir = "./model/loss_v1"
