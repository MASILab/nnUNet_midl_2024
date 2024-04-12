from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
import torch.nn as nn
import numpy as np
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from pdb import set_trace


class Fuse_Average_Logits(nn.Module):

    def __init__(self, t2_model, t1_model):
        super(Fuse_Average_Logits, self).__init__()
        self.t2_model = t2_model
        self.t1_model = t1_model


    def forward(self, x):

        # print("input shape", x.shape)
        # assert x.shape[1] == 2

        # unconcat x into two 
        t2_x = x[:, 0:1, :, :, :]
        t1_x = x[:, 1:2, :, :, :]


        #set_trace()
        # print("t2 input after split", t2_x.shape)
        # print("t1 input after split", t1_x.shape)


        # output during training is a list of deep supervisoion - multi-level output
        t2_logits = self.t2_model(t2_x)
        t1_logits = self.t1_model(t1_x)

        #set_trace()
        
        if self.training:
            # averaging the highest resolution logits
            fused_logits = (t2_logits[0] + t1_logits[0]) / 2.0


            #set_trace()
            return fused_logits, t2_logits[1:], t1_logits[1:]

        # inference
        else:
            print("not training")
            #during inference we are not training and the logits are just tensors of batch x channel x image dims
            # all we are doing here is removing the indexing, because it is not a list of multi-level tensors, but just a single tensor
            fused_logits = (t2_logits + t1_logits) / 2.0


            #set_trace()
            # only return the highest resolution output averaged during inference
            return fused_logits

class nnUNetTrainer_Average_Logits(nnUNetTrainer):


    # Rewriting the network architecture
    @classmethod
    def build_network_architecture( cls, plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision) -> nn.Module:
        t2_network = get_network_from_plans(
            plans_manager, dataset_json, configuration_manager,
            num_input_channels, deep_supervision=enable_deep_supervision)

        t1_network = get_network_from_plans(
            plans_manager, dataset_json, configuration_manager,
            num_input_channels, deep_supervision=enable_deep_supervision)

        # change the shape of the first layer on both networks.
        # becuase we are using base multimodal models that expect 2 channels, we need to change it to 1 channel
        # print(t1_network.encoder)
        t1_conv_block = t1_network.encoder.stages[0][0].convs[0]
        setattr(t1_conv_block, "conv", nn.Conv3d(1, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))) # reassign to 1 input channel for a single modality
        t1_conv_block.all_modules[0] = t1_conv_block.conv
        
        t2_conv_block = t2_network.encoder.stages[0][0].convs[0]
        setattr(t2_conv_block, "conv", nn.Conv3d(1, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))) # reassign to 1 input channel for a single modality
        t2_conv_block.all_modules[0] = t2_conv_block.conv
        
        # setattr(conv_block.all_modules[0], )


        # fuse the models 
        network = Fuse_Average_Logits(t2_model=t2_network, t1_model=t1_network)

        return network   

    def compute_loss(self, output, target):
        fused_logits, t2_logits, t1_logits = output

        loss1 = self.loss([fused_logits] + t2_logits, target)
        loss2 = self.loss([fused_logits] + t1_logits, target)



        # print(loss1, loss2)
        return loss1 + loss2


    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            self.network.module.t2_model.decoder.deep_supervision = enabled
            self.network.module.t1_model.decoder.deep_supervision = enabled

        else:
            self.network.t2_model.decoder.deep_supervision = enabled
            self.network.t1_model.decoder.deep_supervision = enabled












