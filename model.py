# Feel free to change / extend / adapt this source code as needed to complete the homework, based on its requirements.
# This code is given as a starting point.
#
# REFEFERENCES
# The code is partly adapted from pytorch tutorials, including https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# ---- hyper-parameters ----
# You should tune these hyper-parameters using:
# (i) your reasoning and observations, 
# (ii) by tuning it on the validation set, using the techniques discussed in class.
# You definitely can add more hyper-parameters here.
batch_size = 1
max_num_epoch = 100
hps = {'lr':0.008}

layer_count = 2
kernel_count = 16 # select 1 if layer count is equal to 1

batch_normalization = False
tan_h = True

early_stop_active = True
early_stop_e = 0.0001 
import torch
# ---- options ----
DEVICE_ID = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG_DIR = 'checkpoints'
VISUALIZE = False # set True to visualize input, prediction and the output from the last batch
LOAD_CHKPT = True

TEST = False
TRAIN = True

# --- imports ---

import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import hw3utils
torch.multiprocessing.set_start_method('spawn', force=True)


seed = 483
torch.manual_seed(seed)
# Set a random seed for NumPy
np.random.seed(seed)

# ---- utility functions -----
def get_loaders(batch_size,device):
    data_root = 'ceng483-hw3-dataset' 
    train_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'train'),device=device)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'val'),device=device)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)


    # Note: you may later add test_loader to here.
    return train_loader, val_loader


def get_test_loader():
    test_root = 'test'
    test_set = hw3utils.HW3ImageFolder(root=os.path.join(test_root),device=device)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # Note: you may later add test_loader to here.
    return test_loader


def margin_loss_12(predictions, targets, threshold=12):
    
    predictions = (predictions + 1) * 127.5
    targets = (targets + 1) * 127.5

    
    absolute_difference = torch.abs(predictions - targets)

    
    loss = torch.clamp(absolute_difference - threshold, min=0.0)

    
    average_loss = torch.mean(loss)


    return average_loss.item() / 255

# ---- ConvNet -----
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, kernel_count, 3, padding=1)
        self.conv2 = nn.Conv2d(kernel_count, kernel_count, 3, padding=1)
        self.conv3 = nn.Conv2d(kernel_count, kernel_count, 3, padding=1)
        self.convf = nn.Conv2d(kernel_count, 3, 3, padding=1)

        self.batch_normal = nn.BatchNorm2d(kernel_count)
        self.batch_normal_final = nn.BatchNorm2d(3)

    def forward(self, grayscale_image):
        # apply your network's layers in the following lines:
        x = grayscale_image
        if layer_count >= 2:

            x = self.conv1(x)

            if batch_normalization:
                x = self.batch_normal(x)

            x = F.relu(x)  
        if layer_count >= 3:

            x = self.conv2(x)
            
            if batch_normalization:
                x = self.batch_normal(x)

            x = F.relu(x)             
        if layer_count >= 4:

            x = self.conv3(x)
            
            if batch_normalization:
                x = self.batch_normal(x)

            x = F.relu(x)




        x = self.convf(x)
        
        if batch_normalization:
            x = self.batch_normal_final(x)

        if tan_h:
            x = F.tanh(x)


        return x
    
    
# ---- training code -----
device = torch.device(DEVICE_ID)
print('device: ' + str(device))
net = Net().to(device=device)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=hps['lr'])
train_loader, val_loader = get_loaders(batch_size,device)

if LOAD_CHKPT:
    print('loading the model from the checkpoint')
    saved_state = torch.load('final.pt') 
    net.load_state_dict(saved_state)
    #model.load_state_dict(os.path.join(LOG_DIR,'checkpoint.pt'))

losses_train = []
losses_validation = []
learnrate = hps['lr']


if TRAIN:
    print('training begins')
    for epoch in range(max_num_epoch):  
        running_loss_train = 0.0 # training loss of the network
        running_loss_val = 0.0 # training loss of the network
        average_loss_train = 0.0
        count_loss_train = 0
        count_loss_validation = 0
        average_loss_validation = 0.0


        for iteri, data in enumerate(train_loader, 0):
            inputs, targets = data # inputs: low-resolution images, targets: high-resolution images.

            optimizer.zero_grad() # zero the parameter gradients

            # do forward, backward, SGD step
            preds = net(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            # print loss
            running_loss_train += loss.item()

            
            

            print_n = 100 # feel free to change this constant
            if iteri % print_n == (print_n-1):    # print every print_n mini-batches
                print('[%d, %5d] network-loss: %.3f' %
                    (epoch + 1, iteri + 1, running_loss_train / 100))
                average_loss_train += running_loss_train / 100
                count_loss_train += 1
                running_loss_train = 0.0
                # note: you most probably want to track the progress on the validation set as well (needs to be implemented)

            if (iteri==0) and VISUALIZE: 
                hw3utils.visualize_batch(inputs,preds,targets)


        running_loss_train = 0.0
        running_loss_val = 0.0
        for iteri, data in enumerate(val_loader, 0):
            inputs, targets = data # inputs: low-resolution images, targets: high-resolution images.

            optimizer.zero_grad() # zero the parameter gradients

            # do forward, backward, SGD step
            preds = net(inputs)
            loss = criterion(preds, targets)
            # loss = margin_loss_12(preds, targets)
            
            running_loss_val += loss.item()
            # running_loss_val += loss


            print_n = 100 # feel free to change this constant
            if iteri % print_n == (print_n-1):    # print every print_n mini-batches
                print('[%d, %5d] network-loss-validation: %.3f' %
                    (epoch + 1, iteri + 1, running_loss_val / 100))
                average_loss_validation += running_loss_val / 100
                count_loss_validation += 1
                running_loss_val = 0.0
                # note: you most probably want to track the progress on the validation set as well (needs to be implemented)




        print("average_loss_train---",average_loss_train/count_loss_train,"----------")
        print("average_loss_validation---",average_loss_validation/count_loss_validation,"----------")
        losses_train.append(average_loss_train/count_loss_train)
        losses_validation.append(average_loss_validation/count_loss_validation)





        print('Saving the model, end of epoch %d' % (epoch+1))
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        torch.save(net.state_dict(), os.path.join(LOG_DIR,f'Weights-{layer_count}-{kernel_count}-{learnrate}-t-{tan_h}-b-{batch_normalization}.pt'))
        hw3utils.visualize_batch(inputs,preds,targets,os.path.join(LOG_DIR,f'Images-{layer_count}-{kernel_count}-{learnrate}.png'))


        # if epoch > 5 and losses_validation[-2] <= losses_validation[-1] + early_stop_e and losses_validation[-3] <= losses_validation[-2] + early_stop_e:
        #     break

        # if early_stop_active and epoch > 5 and (losses_validation[-4] + losses_validation[-3] + losses_validation[-2])/3 < losses_validation[-1] + early_stop_e:
        #     break 
        if early_stop_active and epoch > 6 and (losses_validation[-6] + losses_validation[-5] + losses_validation[-4] + losses_validation[-3] + losses_validation[-2])/5 < losses_validation[-1] + early_stop_e:
            break 


    print('Finished Training')
    plt.show()
    # Plot both training and validation losses
    if kernel_count == 1:
        kernel_count = 3
        
    plt.title(f'Layer: {layer_count} Kernel Count: {kernel_count} {hps} TanH: {tan_h} BatchNormalization: {batch_normalization}')

    # if tan_h and batch_normalization:
    #     plt.title(f'Layer: {layer_count} Kernel Count: {kernel_count} {hps} TanH: {tan_h} BatchNormalization: {batch_normalization}')
    # elif tan_h:
    #     plt.title(f'Layer: {layer_count} Kernel Count: {kernel_count} {hps} TanH: {tan_h}')
    # elif batch_normalization:
    #     plt.title(f'Layer: {layer_count} Kernel Count: {kernel_count} {hps} BatchNormalization: {batch_normalization}')
    # else:
    #     plt.title(f'Layer: {layer_count} Kernel Count: {kernel_count} {hps}')

    plt.plot(losses_train[1:], label='Training Loss')  # Exclude the first value
    plt.plot(losses_validation[1:], label='Validation Loss')  # Exclude the first value
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()



    # if tan_h and batch_normalization:
    #     plot_filename = os.path.join(LOG_DIR, f'{layer_count}-{kernel_count}-{learnrate}-t-{tan_h}-b-{batch_normalization}.png')
    # elif tan_h:
    #     plot_filename = os.path.join(LOG_DIR, f'{layer_count}-{kernel_count}-{learnrate}-t-{tan_h}.png')
    # elif batch_normalization:
    #     plot_filename = os.path.join(LOG_DIR, f'{layer_count}-{kernel_count}-{learnrate}-b-{batch_normalization}.png')
    # else:
    #     plot_filename = os.path.join(LOG_DIR, f'{layer_count}-{kernel_count}-{learnrate}.png')

    plot_filename = os.path.join(LOG_DIR, f'{layer_count}-{kernel_count}-{learnrate}-t-{tan_h}-b-{batch_normalization}.png')

    # plot_filename = os.path.join(LOG_DIR, f'trainloss.png')

    plt.savefig(plot_filename)

    plt.show()



if TEST:
    test_loader = get_test_loader()

    results_array = np.zeros((100, 80, 80, 3), dtype=np.float32)
    
    # Iterate through the test set
    for iteri, data in enumerate(test_loader, 0):
        inputs, targets = data

        
        # Get predictions from your model
        preds = net(inputs)

        
        print(f'Iteration {iteri}')
        
        
        preds_array = preds[0].cpu().detach().numpy().transpose(1, 2, 0)
        
        
        results_array[iteri] = preds_array
        print(preds_array.shape)
        if iteri == 99:
            break





    results_array = ((results_array+1) * (255/2)).astype(np.uint8)

    # Save the results array to a file
    np.save('estimations_test.npy', results_array)



    


