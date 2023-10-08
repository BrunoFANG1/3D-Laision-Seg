#!/bin/bash
#SBATCH -A rsteven1_gpu
#SBATCH --job-name=3D_Unet_NewData
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --export=ALL
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryi5@jh.edu
#SBATCH --mem-per-cpu=8G   # memory per cpu-core

ml python
source /home/ryi5/StrokeAI_S/bin/activate

pip install simpleitk
python -c"import torch; device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu'); print(device); print('filename: AttenUnet_1M_KpTrain_NewData.py');print('lr:0.0001, reduction');print('train_date:0614');print('Loss Function: BCELoss')"

nvidia-smi

python AttenUnet_4M_KpTrain_NewData.py --mode "train" \
                --model_save_name "Att_0811_NewData"\
                --train_data_dir "/home/ryi5/New_CT_Modified_Dataset/images" \
                --label_data_dir "/home/ryi5/New_CT_Modified_Dataset/labels \
		--model_save_dir "/home/ryi5/New_Codes/model_save"\
                --lr 0.0001\
		--loss_function "BCEFocal"\
                --epochs 250\
                --bs 4\
                --tensorboard_save_dir "/home/ryi5/New_Codes/tensor_save"\
                --Resume "False"\
                --model_path "None"
