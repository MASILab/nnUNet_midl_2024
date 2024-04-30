from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
import torch.nn as nn
import numpy as np
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

import copy
import pdb


class Fuse_Conv_0(nn.Module):

    def __init__(self, t2_model):
        super(Fuse_Conv_0, self).__init__()
        self.t2_model = t2_model

        # deep copy conv 0 from t2 encoder as a T1 head
        self.t1_head = copy.deepcopy(self.t2_model.encoder.stages[0][0])
        self.decoder = self.t2_model.decoder


    def encode(self, x):
        t2_x = x[:, 0:1, :, :, :]
        t1_x = x[:, 1:2, :, :, :]
        x = t2_x
        
        ret = []
        for i, stage in enumerate(self.t2_model.encoder.stages):


            x = stage(x)

            # we only concatenate on stage 1, which comes after conv 0
            if i == 0:
                feat_t1 = self.t1_head(t1_x)
                x = torch.cat((x, feat_t1), dim=1) # concat on hcannels conv0 output for both T1 and T2 heads
                
            ret.append(x)
        
        output = ret if self.t2_model.encoder.return_skips else ret[-1]


        return output


    def forward(self, x):

        skips = self.encode(x)
        #pdb.set_trace()
        return self.decoder(skips)



class nnUNetTrainer_Conv_0(nnUNetTrainer):


    # Rewriting the network architecture
    @classmethod
    def build_network_architecture( cls, plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision) -> nn.Module:
        t2_network = get_network_from_plans(
            plans_manager, dataset_json, configuration_manager,
            num_input_channels, deep_supervision=enable_deep_supervision)


        # change the shape of the first layer on both networks.
        # becuase we are using base multimodal models that expect 2 channels, we need to change it to 1 channel
        t2_conv_block = t2_network.encoder.stages[0][0].convs[0]
        setattr(t2_conv_block, "conv", nn.Conv3d(1, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))) # reassign to 1 input channel for a single modality
        t2_conv_block.all_modules[0] = t2_conv_block.conv
        
        
        
        # double the input channels to the layer after conv 0 to accept concatenated features from two modalitieis
        t2_conv_block_stage_1 = t2_network.encoder.stages[1][0].convs[0]
        setattr(t2_conv_block_stage_1, "conv", nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))) # reassign to 1 input channel for a single modality
        t2_conv_block_stage_1.all_modules[0] = t2_conv_block_stage_1.conv
        


        # change the input channel shape of the first layer of the last decoder block (upcat 1 in monai) to accept skip connections
        t2_decoder_conv_last_block = t2_network.decoder.stages[-1].convs[0]
        setattr(t2_decoder_conv_last_block, "conv", nn.Conv3d(96, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1) ) ) 
        t2_decoder_conv_last_block.all_modules[0] = t2_decoder_conv_last_block.conv



        # fuse the models 
        network = Fuse_Conv_0(t2_model=t2_network)

        # pdb.set_trace()

        return network   



