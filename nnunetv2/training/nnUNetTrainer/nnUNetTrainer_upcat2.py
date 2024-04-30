from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
import torch.nn as nn
import numpy as np
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

import copy
import pdb
from torch import autocast, nn
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss


# This is modified from the decoder forward function which is defined in a different repo
# https://github.com/MIC-DKFZ/dynamic-network-architectures/blob/main/dynamic_network_architectures/building_blocks/unet_decoder.py#L14
# note that i have replaced self arg with obj, because this function is defined outide of the class.
# so that means we are explicitly passing the object 
def modified_forward(decoder, t1_decoder, skips, t1_skips):
    """
    we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
    :param skips:
    :return:
    """

    ##pdb.set_trace()

    # skips are the predictions from each level of the encoder
    # In this model, there are 7 eleements in skips, it is a list of tensors
    # The higehse resolution prediction, is at index 0, the lowest res is at index -1
    lres_input = skips[-1]
    t1_lres_input = t1_skips[-1]

    seg_outputs = []
    t1_seg_outputs = []
    
    # pdb.set_trace()
    # predict at each block of the decoder
    for s in range(len(decoder.stages)):


        # before and at fusion point, need preditions from both t2 and t1 models
        if s <= 4:
            # upsample the skip connection (prediction from a blcok in the encoder)
            x = decoder.transpconvs[s](lres_input)
            t1_x = t1_decoder.transpconvs[s](t1_lres_input)


            # concatenate the skip connection with the spatially upsampled current input 
            x = torch.cat((x, skips[-(s+2)]), 1)
            t1_x = torch.cat((t1_x, t1_skips[-(s+2)]), 1)

            # make a prediction on the current input (input concat with skip)
            x = decoder.stages[s](x)
            t1_x = t1_decoder.stages[s](t1_x)


        # After fusion point, only make t2 pred
        # And join in the t1 skips because this is the last block
        else:

            ###################
            # does t1_x need to be set to None here?
            t1_x = None
            ###################

            # upsample the skip connection (prediction from a blcok in the encoder)
            x = decoder.transpconvs[s](lres_input)

            # concatenate the skip connection with the spatially upsampled current input 
            x = torch.cat((x, 
                           skips[-(s+2)],
                           t1_skips[-(s+2)] 
                           ), 1)

            # make a prediction on the current input (input concat with skip)
            x = decoder.stages[s](x)
            #print("finished fifth layer")



        # run the 1x1 convolutions
        if s <= 4:
            # when training with deep supervision, we need to get a segmentation map (1x1 conv of logits)
            # this happends at every stage in the decoder
            # deep supervision is basically just a multi-level/hierarchical loss in the decoder
            if decoder.deep_supervision:
                # we keep the segmentation outputs in the list here 
                seg_outputs.append(decoder.seg_layers[s](x))
                t1_seg_outputs.append(t1_decoder.seg_layers[s](t1_x))


            # when not using deep supervision, we still need to run a 1x1 convolution
            # over the logits from the final layer to get the segmentation output
            # that is what happens here
            elif s == (len(decoder.stages) - 1):
                seg_outputs.append(decoder.seg_layers[-1](x))
                t1_seg_outputs.append(t1_decoder.seg_layers[-1](t1_x))


        # fuse
        if s == 4:
            x = torch.cat((x, t1_x), dim=1)

        #print("x shape",x.shape)

        # run the 1x1 convolutions
        # same as above but only on t2 model
        if s > 4:
            if decoder.deep_supervision:
                seg_outputs.append(decoder.seg_layers[s](x))

            elif s == (len(decoder.stages) - 1):
                seg_outputs.append(decoder.seg_layers[-1](x))


        # get ready for next stage of the decoder
        lres_input = x
        t1_lres_input = t1_x


    # invert seg outputs so that the largest segmentation prediction is returned first
    seg_outputs = seg_outputs[::-1]
    t1_seg_outputs = t1_seg_outputs[::-1]


    if decoder.training:
        # when not using deep supervision, we just grab the highest resolution output (a tensor)
        # from the list of tensors
        if not decoder.deep_supervision:
            r = seg_outputs[0] 
            # t1_r = t1_seg_outputs[0]
            t1_r = seg_outputs[0] # make this the fused output when not doing deep supervision, but honestly i dont think this ever gets used



        # when using deep supervision, we need to return the list of segmentations from each level of the decoder
        else:
            r = seg_outputs
            t1_r = t1_seg_outputs

            # pdb.set_trace()

            # prepend the t2 (fused) high res pred to the t1 preds before the loss is computed
            # this syntax is list concatenation, not addition
            t1_r = [ r[0] ] + t1_r

    else:
        r = seg_outputs[0] 
        return r

    #print("Check before computing loss")
    #pdb.set_trace()

    return r, t1_r



class Fuse_Upcat_2(nn.Module):

    def __init__(self, t2_model):
        super(Fuse_Upcat_2, self).__init__()
        self.t2_model = t2_model

        self.t1_head = copy.deepcopy(self.t2_model.encoder)
        
        self.decoder = self.t2_model.decoder
        self.t1_decoder = copy.deepcopy(self.t2_model.decoder)

    def encode(self, x):

        # pdb.set_trace()

        # separate modalities
        t2_x = x[:, 0:1, :, :, :]
        t1_x = x[:, 1:2, :, :, :]
        
        # t2 is the main modality
        x = t2_x
            
        ret = []
        t1_ret = []
        # loop through t2 encoder
        for i, stage in enumerate(self.t2_model.encoder.stages):

            x = stage(x) 
            t1_x = self.t1_head.stages[i](t1_x)

            ret.append(x)
            output = ret if self.t2_model.encoder.return_skips else ret[-1]

            t1_ret.append(t1_x)
            t1_output = t1_ret if self.t2_model.encoder.return_skips else t1_ret[-1]
        


        return output, t1_output


    def forward(self, x):

        skips, t1_skips = self.encode(x)

        # pdb.set_trace()

        if self.t2_model.training:
            # because the decoder's forward pass has been redefined to be modified_forward() 
            # which is defined at the top of this file
            # we need to explicitly pass the decoder, and cannot implictly be using self
            # this is because the modified forward function is defined outside of the class
            #return self.decoder(self.decoder, skips, t1_skips)
            t2_preds, t1_preds = self.decoder(  self.decoder,
                                                self.t1_decoder,
                                                skips, 
                                                t1_skips)

            return t2_preds, t1_preds
        
        else:
            preds = self.decoder(  self.decoder,
                                    self.t1_decoder,
                                    skips, 
                                    t1_skips)

            return preds
        
        


class nnUNetTrainer_Upcat2(nnUNetTrainer):


    # Rewriting the network architecture
    @classmethod
    def build_network_architecture( cls, plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision) -> nn.Module:
        t2_network = get_network_from_plans(
            plans_manager, dataset_json, configuration_manager,
            num_input_channels, deep_supervision=enable_deep_supervision)

        #pdb.set_trace()

        # change the shape of the first layer on both networks.
        # becuase we are using base multimodal models that expect 2 channels, we need to change it to 1 channel
        t2_conv_block = t2_network.encoder.stages[0][0].convs[0]
        setattr(t2_conv_block, "conv", nn.Conv3d(1, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))) # reassign to 1 input channel for a single modality
        t2_conv_block.all_modules[0] = t2_conv_block.conv
        




        # change the input channel shape of the first layer of the last decoder block (upcat 1 in monai) to accept skip connections
        t2_decoder_conv_last_block = t2_network.decoder.stages[-1].convs[0]
        setattr(t2_decoder_conv_last_block, "conv", nn.Conv3d(96, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)) ) 
        t2_decoder_conv_last_block.all_modules[0] = t2_decoder_conv_last_block.conv




        # double the input channels to the transpose convolution after the fusion block in the decoder to handle the concatenated featuremaps
        t2_network.decoder.transpconvs[5] = nn.ConvTranspose3d(128, 32, kernel_size=(1, 2, 2), stride=(1, 2, 2))




        # fuse the models 
        network = Fuse_Upcat_2(t2_model=t2_network)



        ##pdb.set_trace()

        # modify the decoder object's forward pass so that we can perform fusion in the decoder
        # this function is at the top of this file
        network.decoder.forward = modified_forward

        ##pdb.set_trace()


        return network   


    def compute_loss(self, output, target):
        # fused_logits is the fused highest reolsuiton output (deep supervision)
        # t2_Logits and t1_logits are lists of the rest of the resolution outputs (highest res was removed)
        t2_preds, t1_preds = output

        # we use [] notation and the + symbol as concatenation of list elements
        # computing the deep supervision loss expects a list of reosolution predictions (tensors)
        # fused_logits gets prepended because it is the highest resolution and highest reosolution goes at beginning
        # to handle the deep supervision loss for both model halves (t2 and t1), 
        # we compute a loss for the T2 half using the fused logits as the highest res
        # and we do the same for the t1 half
        # then we sum the losses 
        loss1 = self.loss(t2_preds, target)
        loss2 = self.loss(t1_preds, target)



        # print(loss1, loss2)
        return loss1 + loss2


    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            #print("self.is_ddp == True has not been defined")
            exit()
            #self.network.module.t2_model.decoder.deep_supervision = enabled
            #self.network.module.t1_model.decoder.deep_supervision = enabled

        else:
            self.network.t2_model.decoder.deep_supervision = enabled
            self.network.t1_decoder.deep_supervision = enabled





    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.forward_pass(data)    # channge
            del data
            pdb.set_trace()
            l = self.compute_loss(output, target)


            # output = self.network(data)
            # del data
            # l = self.loss(output, target)

        #pdb.set_trace()
        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            #output = output[0]

            # output is t2 and t1 preds. 
            # t1 preds are only needed for the deep supervision loss
            # here we only care about the t2 preds because his had the fusion. 
            output = output[0][0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}



