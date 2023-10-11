import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
import tqdm
from torch.utils.data import DataLoader
import CTDataset
import torch.optim as optim
import torch.utils.data as data

def get_args( args : list) -> dict:

    parser = argparse.ArgumentParser(description ='3dUnet command line argument parser')
    parser.add_argument('--mode',
                        help = 'the action you want to do [train],[test]',
                        type = str,
                        choices =["train", "predict"],
                        required = True)

    parser.add_argument('--model_save_name',
                        help = 'the training result model name',
                        type = str)

    parser.add_argument('--train_data_dir',
                        help = 'directory contains training data',
                        type = str)
    parser.add_argument('--label_data_dir',
                        help = 'directory contains label data',
                        type = str)

    parser.add_argument('--model_save_dir',
                        help = 'directory to save the model checkpoint',
                        type = str)
    parser.add_argument('--lr',
                        help = 'learning rate',
                        type = float)
    parser.add_argument('--loss_function',
                        help ='the loss function you want use',
                        type = str,
                        choices =["BCELoss", "CrossEntropy","BCEFocal","CrossEntropyFocal","DiceLoss"],
                        required  = True)
    parser.add_argument('--epochs',
                        help = 'Epochs for training',
                        type = int)
    parser.add_argument('--bs',
                        help = 'batch size',
                        type = int)
    parser.add_argument('--tensorboard_save_dir', 
                        help = 'The directory to save the tensorbroad data',
                        type = str)
    parser.add_argument('--Resume',
                        help = 'The training resume or not',
                        choices =["True", "False"],
                        type = str)
    parser.add_argument('--model_path',
                        help = 'the input training model save path',
                        type = str)
    options = vars(parser.parse_args())
    return options


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


torch.autograd.set_detect_anomaly(True)

def train(model, optimizer, loss_metric, start_epoch, lr, epochs, train_dataloader, val_dataloader, test_dataloader, tensor_save,**kwargs):
    net = model
    if torch.cuda.device_count()>1:
        print('Lets use',torch.cuda.device_count(),'GPU!')
    writer = SummaryWriter(tensor_save)
    train_loss_list = []
    val_loss_list = []
    train_name_list =[]
    val_name_list =[]
    test_name_list =[]
    # step_sche = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.1)
    print("Start Training")
    print(print_network(net))

    for epoch in tqdm(range(epochs)):

        print(f"==========Epoch:{epoch}==========lr:{lr}========{epoch}/{epochs}")
        loss_sum = 0
        dice_sum = 0
        train_batch_count = 0
        val_batch_count = 0
             
        for ctid, mriid, img, mask in train_dataloader:
            img, mask = img.float(), mask.float()
            img, mask = img, mask.to("cuda:1")
            net.train()
            optimizer.zero_grad()

            # forward prop
            output = net.forward(img)
            print("output shape: ",output.shape)
            print("mask shape: ",mask.shape)
            loss = loss_metric(output, mask)          #######?????

            # backward prop
            loss.backward()
            optimizer.step()

            print("train_loss: ", loss)
            loss_sum += loss
            train_name_list.append(ctid)
            train_batch_count = train_batch_count+1


        train_loss = loss_sum.item()/train_batch_count
        writer.add_scalar("train loss",float(train_loss),epoch)
        train_loss_list.append(train_loss)
            

        with torch.no_grad():
            loss_sum = 0
            dice_sum = 0
            for ctid, mriid, img, mask in val_dataloader:
                img, mask = img.float(), mask.float()
                img, mask = img.cuda(), mask.cuda()
                net.eval()
                output = net.forward(img)
                loss = loss_metric(output, mask)
                print("val_loss: ", loss)
                loss_sum += loss 
                val_name_list.append(ctid)
                val_batch_count = val_batch_count+1
            print("val_batch_count: ",val_batch_count)
            val_loss = loss_sum.item()/val_batch_count
            val_loss_list.append(val_loss)
            writer.add_scalar("valid loss",float(val_loss),epoch)


def main(args):

    arguments = get_args(args.argv)
    MODE = arguments.get('mode')
    TRAIN_DIR = arguments.get('train_data_dir')
    LABEL_DIR = arguments.get('label_data_dir')
    LOSS_FCN = arguments.get('loss_function')
    TENSOR = arguments.get('tensorboard_save_dir')

    if MODE == 'train':
        BESTMODEL_NAME = arguments.get('model_save_name')
        MODEL_SAVE_DIR = arguments.get('model_save_dir')
        LEARNING_RATE = arguments.get('lr')
        EPOCHS = arguments.get('epochs')
        BATCH_SIZE = arguments.get('bs')
        RESUME_TIP = arguments.get('Resume')
        MODEL_PATH  = arguments.get('model_path')
        # early_stopping = EarlyStopping(MODEL_SAVE_DIR)

        # Dataset Loading
        CT_path = TRAIN_DIR
        MRI_path = LABEL_DIR
        train_set = CTDataset(CT_image_root = CT_path, MRI_label_root = MRI_path)
        train_set_size = int(train_set.__len__()*0.8)
        test_set_size = len(train_set) - train_set_size
        train_set, test_set = data.random_split(train_set, [train_set_size, test_set_size])
        train_set_size1 = int(len(train_set)*0.8)
        valid_set_size = train_set_size - train_set_size1
        train_set, valid_set = data.random_split(train_set, [train_set_size1, valid_set_size])
        print("Train data set:",len(train_set))
        print("Valid data set:", len(valid_set))
        print("Test data set:", len(test_set))
        Batch_Size = BATCH_SIZE
        train_loader = DataLoader(dataset = train_set, batch_size = Batch_Size,shuffle=True)
        valid_loader = DataLoader(dataset = valid_set, batch_size = Batch_Size,shuffle=True)
        test_loader = DataLoader(dataset = test_set, batch_size = Batch_Size)
        

        # Set up model, optimizer, loss_function
        start_epoch = 0
        model = Att_Unet()
        model = model.cuda()
        optimizer = torch.optim.AdamW(net.parameters(), lr = lr,weight_decay = 1e-5)
        loss_fcn = FocalLoss(to_onehot_y = False)

        if RESUME_TIP == 'True':
            start_epoch = checkpoint['epoch']
            path_checkpoint = MODEL_PATH
            checkpoint = torch.load(path_checkpoint)
            model.load_state_dict(checkpoint)
            optimizer = optim.Adam
            optimizer.load_state_dict(checkpoint['optimizer'])

        print("Start training")
        output = train(model = model, 
                    optimizer = optimizer, 
                    loss_metric = loss_fcn, 
                    start_epoch = start_epoch,
                    lr =LEARNING_RATE, 
                    epochs = EPOCHS, 
                    train_dataloader = train_loader,
                    val_dataloader = valid_loader,
                    test_dataloader = test_loader,
                    tensor_save = TENSOR)    
        print('Training Finished')


if __name__ == '__main__':
    args = get_args()
    args = args.parse_args()
    main(args)