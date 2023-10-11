from CTDataset import CTDataset
import torch
from loss import DiceLoss
from model.Unet import UNet3D
from torch.utils.data import DataLoader, random_split

def main():
    # construct dataset
    train_set = CTDataset(CT_image_root = "/home/bruno/xfang/dataset/images/", MRI_label_root = "/home/bruno/xfang/dataset/labels/")
    print(len(train_set))
    train_dataloader = DataLoader(dataset = train_set, batch_size = 1, shuffle=True)
    
    model = UNet3D(in_channels=1,
                    out_channels=1,
                      )
    loss = DiceLoss()
    lr = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr,weight_decay = 1e-5) 
    epochs = 10
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in range(epochs):
        for ctid, mriid, img, mask in train_dataloader:
            img, mask = img.to(device), mask.to(device)
            print("one batch starts")
            model.train()
            output = model.forward(img)
            loss_ = loss(output, mask)
            loss_.backward()
            optimizer.step()

            print(loss_)

if __name__ == '__main__':
    main()