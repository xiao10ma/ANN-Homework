import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from net_utils import save_model, load_model
import argparse
import face_dataset
from model import AlexNet, VGG, ResNet50
from tqdm import tqdm
import uuid
from sklearn.metrics import recall_score, f1_score
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 400
LR= 0.00013     # AlexNet: 0.00013, VGG: 1e-05, ResNet: 0.0002

def train(network, tb_writer, args):
    train_config = {'data_root' : args.data_source, 'split' : 'train', 'image_size' : (256, 256)}
    train_dataset = face_dataset.faceDataset(**train_config)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8)

    test_config = {'data_root' : args.data_source, 'split' : 'test', 'image_size' : (256, 256)}
    test_dataset = face_dataset.faceDataset(**test_config)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=True, num_workers=8)  

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(network.parameters(), lr=LR)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # Load trained model
    begin_epoch, global_step = load_model(network, optimizer, args.model_path)
    # begin_epoch, global_step = 0, 0

    for epoch in tqdm(range(begin_epoch, EPOCH)):
        for iter, batch in enumerate(train_loader):
            iter_start.record()
            # batch[0]: img, batch[1]:gt_label
            images, gt_label = batch[0], batch[1]
            images, gt_label = images.to(device), gt_label.to(device)
            optimizer.zero_grad()
            pred_label = network(images)
            loss = criterion(pred_label, gt_label)
            loss.backward()
            optimizer.step()

            iter_end.record()
            torch.cuda.synchronize()
            if (iter % 10 == 0):
                global_step += 1
                acc, recall, macro_f1 = evaluate(network, test_loader)
                training_report(tb_writer, global_step, loss, iter_start.elapsed_time(iter_end), acc, recall, macro_f1)
        if (epoch + 1) % args.save_ep == 0:
            save_model(network, optimizer, args.model_path, epoch, global_step)
        if (epoch + 1) % args.save_latest_ep == 0:
            save_model(network, optimizer, args.model_path, epoch, global_step, last=True)
    
    acc, recall, macro_f1 = evaluate(network, test_loader)
    print(f"Accuracy: {acc}  Recall: {recall}  Macro-F1: {macro_f1}")



def evaluate(network, test_loader):
    all_pred_labels = []
    all_gt_labels = []
    
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            # batch[0]: img, batch[1]: gt_label
            images, gt_label = batch[0], batch[1]
            images, gt_label = images.to(device), gt_label.to(device)
            output = network(images)
            
            # Convert output probabilities to predicted labels
            pred_label = torch.argmax(output, dim=1)
            gt_label = torch.argmax(gt_label, dim=1)
            
            # Store predictions and actual labels for later analysis
            all_pred_labels.extend(pred_label.cpu().numpy())
            all_gt_labels.extend(gt_label.cpu().numpy())
    
    # Calculate metrics
    accuracy = (np.array(all_pred_labels) == np.array(all_gt_labels)).mean()
    recall = recall_score(all_gt_labels, all_pred_labels, average='macro')
    macro_f1 = f1_score(all_gt_labels, all_pred_labels, average='macro')
    
    return accuracy, recall, macro_f1

def prepare_output_and_logger(args):    
    if not args.record_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.record_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.record_path))
    os.makedirs(args.record_path, exist_ok = True)
    with open(os.path.join(args.record_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(argparse.Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.record_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iter, loss, elapsed, acc, recall, macro_f1):
    if tb_writer:
        tb_writer.add_scalar('loss' + '  lr: {}  epoch: {}'.format(LR, EPOCH), loss.item(), iter)
        tb_writer.add_scalar('iter_time' + '  lr: {}  epoch: {}'.format(LR, EPOCH), elapsed, iter)
        tb_writer.add_scalar('acc' + '  lr: {}  epoch: {}'.format(LR, EPOCH), acc, iter)
        tb_writer.add_scalar('recall' + '  lr: {}  epoch: {}'.format(LR, EPOCH), recall, iter)
        tb_writer.add_scalar('macro-f1' + '  lr: {}  epoch: {}'.format(LR, EPOCH), macro_f1, iter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Recognition')
    parser.add_argument('-s', '--data_source', default='./data', type=str)
    parser.add_argument('-r', '--random', default=True, action='store_false')
    parser.add_argument('--record_path', default='./output/AlexNet/AlexNet-lr_{}epoch_{}'.format(LR, EPOCH), type=str)
    parser.add_argument('--model_path', default='./trained_model/AlexNet', type=str)
    parser.add_argument('--save_ep', default=50, type=int)
    parser.add_argument('--save_latest_ep', default=10)

    args = parser.parse_args()

    tb_writer = prepare_output_and_logger(args)


    network = AlexNet().to(device)
    # network = VGG().to(device)
    # network = ResNet50(5).to(device)


    train(network, tb_writer, args)