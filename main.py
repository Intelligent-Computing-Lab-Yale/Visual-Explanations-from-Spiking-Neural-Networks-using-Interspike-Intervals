
import argparse
import numpy as np
import os.path
import matplotlib.gridspec as gridspec
import skimage.transform
import matplotlib.pyplot as plt
from utils import *
from   torch.utils.data.dataloader import DataLoader
from   torchvision import transforms, datasets
from   model import *


parser = argparse.ArgumentParser(description='Visual explanations from spiking neural networks using inter‑spike intervals', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--pretrainedmodel_pth', default='PATH/TO/MODEL', type=str, help='path for pretrained model')
parser.add_argument('--dataset_pth', default='PATH/TO/DATASET', type=str, help='path for validation dataset')
parser.add_argument('--timesteps',             default=30,    type=float, help='timesteps')
parser.add_argument('--batch_size',            default=1,       type=int,   help='batch size should be 1')
parser.add_argument('--leak_mem',              default=0.99,   type=float, help='leak_mem')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--gamma', default=0.5, type=float, help='float')
parser.add_argument('--target_layer', default=8, type=int, help='target_layer [4, 6, 8] is available')

global args
args = parser.parse_args()

save_model_path = args.pretrainedmodel_pth
save_model_statedict = torch.load(save_model_path)['state_dict']
save_model_accuracy = torch.load(save_model_path)['accuracy1']

# select number of samples for visualization
img_nums = [10, 52]

gamma = args.gamma
num_timestep = args.timesteps
batch_size      = args.batch_size
visual_imagesize = 128
target_layer = args.target_layer


# Normalization function
class normalize(object):
    def __init__(self, mean, absmax):
        self.mean = mean
        self.absmax = absmax
    def __call__(self, tensor):
        for t, m, am in zip(tensor, self.mean, self.absmax):
            t.sub_(m).div_(am)
        return tensor


# Mean and SD are calculuated
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

valdir = os.path.join(args.dataset_pth)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

testloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers)


display_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize((visual_imagesize, visual_imagesize)),
            transforms.ToTensor(),
        ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers)

criterion = nn.CrossEntropyLoss()


#--------------------------------------------------
# Instantiate the SNN model
#--------------------------------------------------
model = SNN_VGG11()


print('Loading Model')

model = torch.nn.DataParallel(model).cuda()
save_model_statedict = torch.load(save_model_path)['state_dict']
cur_dict = model.state_dict()

for key in save_model_statedict.keys():
    if key in cur_dict:
        if (save_model_statedict[key].shape == cur_dict[key].shape):
            cur_dict[key] = save_model_statedict[key]
        else:
            print("Error mismatch")

model.load_state_dict(cur_dict)
model.train()



#--------------------------------------------------
# Extracting heatmap
#--------------------------------------------------
print('********** Extracting heatmap **********')


def getCAM(feature_conv, weight):
    _, nc, h, w = feature_conv.shape
    cam = weight.dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / (np.max(cam) +1e-3)
    return [cam_img]

def getForwardCAM(feature_conv):
    cam = feature_conv.sum(axis =0).sum(axis =0)
    cam = cam - np.min(cam)
    cam_img = cam / (np.max(cam) +1e-3)
    return [cam_img]


cam_dict = {}
show_flag = True


model.eval()

fig, axes = plt.subplots(len(img_nums), 10+1)
gs1 = gridspec.GridSpec(len(img_nums), 10+1)
gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes.

img_idx = 0


for j, (data, data_disp) in enumerate(zip(testloader, display_loader)):

    if show_flag is True:
        if j not in img_nums:
            continue
        if j > np.max(img_nums):
            exit()

    if j % 100 == 0:
        print ('iteration {}/{}'.format(j, len(testloader)))

    model.zero_grad()
    model.module.saved_grad = 0
    model.module.saved_forward = []

    images_disp, labels_disp  = data_disp
    images, labels_cpu = data
    images = images.cuda()
    labels = labels_cpu.cuda()

    output_list  = model(images, target_layer=target_layer)

    if show_flag is True:
        axes[img_idx, 0] = plt.subplot(gs1[img_idx, 0])
        axes[img_idx, 0].axis('off')
        axes[img_idx, 0].imshow(images_disp[0, ...].permute(1, 2, 0))

    process = 0
    time = 0
    cam_save = 0
    overlay_list = []
    previous_spike_time_list = []
    activation_list_value = (model.module.saved_forward)

    for l, activation in enumerate(activation_list_value):
        activation = activation
        previous_spike_time_list.append(activation)
        weight = 0

        for prev_t in range(len(previous_spike_time_list)):
            delta_t = time - previous_spike_time_list[prev_t]* prev_t
            weight +=  torch.exp(gamma * (-1) * delta_t)

        weighted_activation = weight.cuda() * activation
        weighted_activation = weighted_activation.data.cpu().numpy()
        overlay = getForwardCAM(weighted_activation)
        overlay_list.append(overlay[0])

        if show_flag is True:
            if process%3 == 0:
                axes[img_idx, process//3+1 ] = plt.subplot(gs1[img_idx, process//3+1 ])
                axes[img_idx, process//3+1 ].axis('off')
                axes[img_idx, process//3+1].imshow(images_disp[0, ...].permute(1, 2, 0))
                axes[img_idx, process//3+1 ].imshow(
                    skimage.transform.resize(overlay[0], (visual_imagesize, visual_imagesize)), alpha=0.5, cmap='jet')

        process += 1
        time += 1

    cam_dict[j] = overlay_list
    img_idx +=1

if show_flag is True:
    if os.path.isdir('figuresave') is not True:
        os.mkdir('figuresave')
    plt.savefig(os.path.join('figuresave', 'actmap_ly'+str(target_layer)+'.png' ), dpi=600)


