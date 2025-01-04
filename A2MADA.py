from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch import nn
import os

# Gradient inversion layer
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

def extract_layer_outputs(captured_outputs):
    # Extract the output of a specific layer
    layer_24_output = captured_outputs.get(24, None)
    layer_37_output = captured_outputs.get(37, None)
    layer_50_output = captured_outputs.get(50, None)
    layer_75_output = captured_outputs.get(75, None)
    layer_88_output = captured_outputs.get(88, None)
    layer_101_output = captured_outputs.get(101, None)

    return layer_24_output, layer_37_output, layer_50_output, layer_75_output, layer_88_output, layer_101_output

class A2MADA(nn.Module):
    def __init__(self, device):
        super(A2MADA, self).__init__()
        self.da_img_model_6 = DA_Img_WithAtt(512).to(device)
        self.da_img_model_8 = DA_Img_WithAtt(1024).to(device)
        self.da_img_model_10 = DA_Img_WithAtt(1024).to(device)
        self.da_ins_model_21 = DAInsHead(128).to(device)
        self.da_ins_model_23 = DAInsHead(256).to(device)
        self.da_ins_model_25 = DAInsHead(512).to(device)

    def forward(self, captured_src, captured_tgt):
        layers_src = extract_layer_outputs(captured_src)
        layers_tgt = extract_layer_outputs(captured_tgt)

        da_img_loss = 0
        da_ins_loss = 0
        da_consist_loss = 0
        attention_loss = 0

        da_img_models = [self.da_img_model_6, self.da_img_model_8, self.da_img_model_10]
        da_ins_models = [self.da_ins_model_21, self.da_ins_model_23, self.da_ins_model_25]

        src_img_consist_outputs = []
        tgt_img_consist_outputs = []
        src_ins_consist_outputs = []
        tgt_ins_consist_outputs = []

        # Compute the image-level domain adaptive loss
        for i, (src_layer, tgt_layer) in enumerate(zip(layers_src[:3], layers_tgt[:3])):
            src_img_rev = ReverseLayerF.apply(src_layer, alpha)
            tgt_img_rev = ReverseLayerF.apply(tgt_layer, alpha)
            src_img_consist = ReverseLayerF.apply(src_layer, -alpha)
            tgt_img_consist = ReverseLayerF.apply(tgt_layer, -alpha)

            da_img_model = da_img_models[i]
            src_grl_output, src_att_maps = da_img_model(src_img_rev)
            tgt_grl_output, tgt_att_maps = da_img_model(tgt_img_rev)

            src_img_consist_output, _ = da_img_model(src_img_consist)
            tgt_img_consist_output, _ = da_img_model(tgt_img_consist)
            src_img_consist_output = src_img_consist_output.sigmoid()
            tgt_img_consist_output = tgt_img_consist_output.sigmoid()

            src_img_consist_outputs.append(src_img_consist_output)
            tgt_img_consist_outputs.append(tgt_img_consist_output)

            # The MSE between each attention map is calculated as the attention alignment loss
            for src_att_map, tgt_att_map in zip(src_att_maps, tgt_att_maps):
                attention_loss += F.mse_loss(src_att_map, tgt_att_map)

            da_source_label = torch.zeros_like(src_grl_output).type(torch.cuda.FloatTensor)
            da_target_label = torch.ones_like(tgt_grl_output).type(torch.cuda.FloatTensor)
            grl_s_loss = F.binary_cross_entropy_with_logits(src_grl_output, da_source_label)
            grl_t_loss = F.binary_cross_entropy_with_logits(tgt_grl_output, da_target_label)
            da_img_loss += grl_s_loss + grl_t_loss
            
        src_img_consist_mean = process_outputs(src_img_consist_outputs)
        tgt_img_consist_mean = process_outputs(tgt_img_consist_outputs)

        # Compute the instance-level domain adaptive loss
        for i, (src_layer, tgt_layer) in enumerate(zip(layers_src[3:], layers_tgt[3:])):
            src_ins_rev = ReverseLayerF.apply(src_layer, alpha)
            tgt_ins_rev = ReverseLayerF.apply(tgt_layer, alpha)
            src_ins_consist = ReverseLayerF.apply(src_layer, -alpha)
            tgt_ins_consist = ReverseLayerF.apply(tgt_layer, -alpha)

            da_ins_model = da_ins_models[i]
            src_ins_rev = F.adaptive_avg_pool2d(src_ins_rev, (1, 1)).squeeze()
            tgt_ins_rev = F.adaptive_avg_pool2d(tgt_ins_rev, (1, 1)).squeeze()

            src_ins_output = da_ins_model(src_ins_rev)
            tgt_ins_output = da_ins_model(tgt_ins_rev)

            src_ins_consist_output = da_ins_model(F.adaptive_avg_pool2d(src_ins_consist, (1, 1)).squeeze()).sigmoid()
            tgt_ins_consist_output = da_ins_model(F.adaptive_avg_pool2d(tgt_ins_consist, (1, 1)).squeeze()).sigmoid()

            src_ins_consist_outputs.append(src_ins_consist_output)
            tgt_ins_consist_outputs.append(tgt_ins_consist_output)

            da_source_label = torch.zeros_like(src_ins_output).type(torch.cuda.FloatTensor)
            da_target_label = torch.ones_like(tgt_ins_output).type(torch.cuda.FloatTensor)
            grl_s_loss = F.binary_cross_entropy_with_logits(src_ins_output, da_source_label)
            grl_t_loss = F.binary_cross_entropy_with_logits(tgt_ins_output, da_target_label)
            da_ins_loss += grl_s_loss + grl_t_loss

        src_ins_consist_mean = torch.mean(torch.cat(src_ins_consist_outputs, dim=0), dim=0, keepdim=True)
        tgt_ins_consist_mean = torch.mean(torch.cat(tgt_ins_consist_outputs, dim=0), dim=0, keepdim=True)

        # Calculating consistency loss
        da_consist_loss = consistency_loss(src_img_consist_mean, src_ins_consist_mean, tgt_img_consist_mean,
                                           tgt_ins_consist_mean, size_average=True)
        da_img_loss /= 6
        da_ins_loss /= 6
        attention_loss = attention_loss / 6

        da_loss = (da_img_loss + da_ins_loss) / 2 + lambda_1 * da_consist_loss + lambda_2 * attention_loss
        return da_loss

class DA_Img_WithAtt(nn.Module):
    def __init__(self, in_channels):
        super(DA_Img_WithAtt, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.cas = nn.ModuleList()  
        if in_channels == 512:
            channels = [512, 256, 128, 1]
        elif in_channels == 1024:
            channels = [1024, 256, 128, 1]
        else:
            raise ValueError("Unsupported in_channels")

        for i in range(len(channels) - 1):
            self.convs.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=1, stride=1))
            self.bns.append(nn.BatchNorm2d(channels[i + 1]))
            if i < len(channels) - 2: 
                self.cas.append(ChannelAttention(channels[i + 1]))
            nn.init.normal_(self.convs[-1].weight, std=0.1)
            nn.init.constant_(self.convs[-1].bias, 0)

    def forward(self, x):
        x = x.to(torch.float32)
        attention_maps = []

        for conv, bn, ca in zip(self.convs, self.bns, self.cas):
            x = conv(x)
            x = bn(x)
            x, attention_map = ca(x)
            attention_maps.append(attention_map)
            x = F.leaky_relu(x)
            
        x = self.convs[-1](x)
        x = self.bns[-1](x)
        x = F.leaky_relu(x)

        img_features = x.view(-1, 1)
        return img_features, attention_maps

class DAInsHead(nn.Module):
    def __init__(self, in_channels):
        super(DAInsHead, self).__init__()
        self.fc1_da = nn.Linear(in_channels, 1024)
        self.fc2_da = nn.Linear(1024, 1024)
        self.fc3_da = nn.Linear(1024, 1)
        for l in [self.fc1_da, self.fc2_da]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.fc3_da.weight, std=0.05)
        nn.init.constant_(self.fc3_da.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.fc2_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc3_da(x)
        return x


def consistency_loss(src_img, src_ins, tgt_img, tgt_ins, size_average=True):
    """
    Calculate consistency loss between image-level and instance-level classifier outputs
    for both source and target domains.
    """
    src_loss = torch.abs(src_img - src_ins)
    tgt_loss = torch.abs(tgt_img - tgt_ins)
    loss = (src_loss + tgt_loss) / 2
    if size_average:
        return loss.mean()
    return loss.sum()

def process_outputs(outputs):
    processed_outputs = [torch.mean(output, dim=0, keepdim=True) for output in outputs]
    overall_mean = torch.mean(torch.cat(processed_outputs, dim=0), dim=0)

    return overall_mean

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        reduced_channels = max(in_channels // reduction_ratio, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        attention = self.sigmoid(out)
        return x * attention, attention
