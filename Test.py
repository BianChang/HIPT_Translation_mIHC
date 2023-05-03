import torch
from SwinVisionTranformer import SwinTransformer

# Load the entire pretrained model
pretrained_model = torch.load(r'D:\Chang_files\workspace\data\MIHC\models\upernet_swin_tiny_patch4_window7_512x512.pth')

state_dict = pretrained_model['state_dict']
for name in state_dict:
    print(name)



#swin_t = SwinTransformer()

# match the layers correctly here
'''
for name, param in swin_t.named_parameters():
    print(name)
    if name in pretrained_model.state_dict():
        print(name, ' exist')
        param.data = pretrained_model.state_dict()[name].data
    else:
        print(name, ' does not exist')
'''

# Freeze the encoder's weights
# for param in swin_t.parameters():
    # param.requires_grad = False