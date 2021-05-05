import os
import pathlib
import torch
from data import PoisonedDataset, load_init_data, create_backdoor_data_loader
from models import BadNet, load_model
from utils.utils import print_model_perform, backdoor_model_trainer
from config import opt


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # create related path
    pathlib.Path("./checkpoints/").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./logs/").mkdir(parents=True, exist_ok=True)

    print("# --------------------------read dataset: %s --------------------------" % opt.dataset)
    train_data, test_data = load_init_data(dataname=opt.dataset, device=device, download=opt.download, dataset_path=opt.datapath)

    print("# --------------------------construct poisoned dataset--------------------------")
    train_data_loader, test_data_ori_loader, test_data_tri_loader = create_backdoor_data_loader(opt.dataset, train_data, test_data, opt.trigger_label, opt.poisoned_portion, opt.batchsize, device)

    print("# --------------------------begin training backdoor model--------------------------")
    basic_model_path = "./checkpoints/badnet-%s.pth" % opt.dataset
    if opt.no_train:
        model = backdoor_model_trainer(
                dataname=opt.dataset,
                train_data_loader=train_data_loader, 
                test_data_ori_loader=test_data_ori_loader,
                test_data_tri_loader=test_data_tri_loader,
                trigger_label=opt.trigger_label,
                epoch=opt.epoch,
                batch_size=opt.batchsize,
                loss_mode=opt.loss,
                optimization=opt.optim,
                lr=opt.learning_rate,
                print_perform_every_epoch=opt.pp,
                basic_model_path= basic_model_path,
                device=device
                )
    else:
        model = load_model(basic_model_path, model_type="badnet", input_channels=train_data_loader.dataset.channels, output_num=train_data_loader.dataset.class_num, device=device)

    print("# --------------------------evaluation--------------------------")
    print("## original test data performance:")
    print_model_perform(model, test_data_ori_loader)
    print("## triggered test data performance:")
    print_model_perform(model, test_data_tri_loader)

if __name__ == "__main__":
    main()
