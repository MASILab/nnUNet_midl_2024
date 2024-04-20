from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
import torch.nn as nn
import numpy as np
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

import copy
import pdb


class Fuse_Down_1(nn.Module):

    def __init__(self, t2_model):
        super(Fuse_Down_1, self).__init__()
        self.t2_model = t2_model
        self.fusion_point = 1

        # deep copy conv 0 from t2 encoder as a T1 head
        # self.t1_head = copy.deepcopy(self.t2_model.encoder.stages[0][0])
        self.t1_head = copy.deepcopy(self.t2_model.encoder.stages[:self.fusion_point + 1])
        self.decoder = self.t2_model.decoder

        # pdb.set_trace()

    def encode(self, x):

        # separate modalities
        t2_x = x[:, 0:1, :, :, :]
        t1_x = x[:, 1:2, :, :, :]
        
        # t2 is the main modality
        x = t2_x
            
        ret = []
        # loop through t2 encoder
        for i, stage in enumerate(self.t2_model.encoder.stages):

            # get the t2 featuremaps
            x = stage(x)

            # skip connections before and at fusion layer need to be t2 concat t1 feature maps
            if i < self.fusion_point:
                t1_x = self.t1_head[i](t1_x)

                skip = torch.cat((x, t1_x), dim=1)

            # fuse featuremaps at fusion point
            elif i == self.fusion_point:
                t1_x = self.t1_head[i](t1_x)

                skip = torch.cat((x, t1_x), dim=1)

                x = torch.cat((x, t1_x), dim=1) # concat on hcannels down 2 output for both T1 and T2 heads


            # after the fusion point, the skip is just the t2 feautre map
            else:
                skip = x


            # these are the skip connections
            # it may also be related to deep supervision
            ret.append(skip)


            output = ret if self.t2_model.encoder.return_skips else ret[-1]


        
        return output


    def forward(self, x):

        skips = self.encode(x)

        # pdb.set_trace()

        return self.decoder(skips)



class nnUNetTrainer_Down1(nnUNetTrainer):


    # Rewriting the network architecture
    @classmethod
    def build_network_architecture( cls, plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision) -> nn.Module:
        t2_network = get_network_from_plans(
            plans_manager, dataset_json, configuration_manager,
            num_input_channels, deep_supervision=enable_deep_supervision)

        # pdb.set_trace()

        # change the shape of the first layer on both networks.
        # becuase we are using base multimodal models that expect 2 channels, we need to change it to 1 channel
        t2_conv_block = t2_network.encoder.stages[0][0].convs[0]
        setattr(t2_conv_block, "conv", nn.Conv3d(1, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))) # reassign to 1 input channel for a single modality
        t2_conv_block.all_modules[0] = t2_conv_block.conv
        

        # pdb.set_trace()

        # double the input channels to the layer after conv 0 to accept concatenated features from two modalitieis
        t2_conv_block_stage_2 = t2_network.encoder.stages[2][0].convs[0]
        #only thing changed here is the input channels. DOuble what would normally be expected 
        setattr(t2_conv_block_stage_2, "conv", nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))) # reassign to 1 input channel for a single modality
        t2_conv_block_stage_2.all_modules[0] = t2_conv_block_stage_2.conv





        # pdb.set_trace()


        # change the input channel shape of the first layer of the last decoder block (upcat 1 in monai) to accept skip connections
        t2_decoder_conv_last_block = t2_network.decoder.stages[-1].convs[0]
        setattr(t2_decoder_conv_last_block, "conv", nn.Conv3d(96, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)) ) 
        t2_decoder_conv_last_block.all_modules[0] = t2_decoder_conv_last_block.conv

        t2_decoder_conv_second_to_last_block = t2_network.decoder.stages[-2].convs[0]
        setattr(t2_decoder_conv_second_to_last_block, "conv", nn.Conv3d(192, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)) ) 
        t2_decoder_conv_second_to_last_block.all_modules[0] = t2_decoder_conv_second_to_last_block.conv


        # pdb.set_trace()s

        # fuse the models 
        network = Fuse_Down_1(t2_model=t2_network)


        return network   



