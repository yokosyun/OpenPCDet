import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
def visualize_activation_histogram(feat, masks, title="activation_histogram"):
    channels = feat.size(1)

    y = feat.permute(1, 0, 2, 3).reshape(channels, -1).cpu().detach().numpy()

    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    plt.title(title)
    plt.xlabel("Activation Values")
    plt.ylabel("Frequency")

    min_val = np.min(y)
    max_val = np.max(y)
    bins = np.linspace(min_val, max_val, 128)
    # plt.ylim(0, 1000)

    kernel_size_h = masks.size(2) / feat.size(2)
    kernel_size_w = masks.size(3) / feat.size(3)

    kernel_size_h = round(kernel_size_h)
    kernel_size_w = round(kernel_size_w)

    mean = feat.mean(dim=[0, 2, 3])

    if kernel_size_h != 1:
        assert kernel_size_h % 2 == 0, "kernel_size_h="+str(kernel_size_h)+"is not supported right now"
        assert kernel_size_w % 2 == 0, "kernel_size_w="+str(kernel_size_w)+"is not supported right now"
        masks = torch.nn.functional.max_pool2d(masks.float(), kernel_size=(int(kernel_size_h),int(kernel_size_w))).bool()


    mask_expanded = masks.expand(-1, feat.size(1), -1, -1)

    mask_expanded = mask_expanded.permute(0, 2, 3, 1)
    feat_permute = feat.permute(0, 2, 3, 1)

    valid_feat = feat_permute[mask_expanded]
    valid_feat = valid_feat.reshape(-1, feat.size(1))

    for i in range(channels):
        mask = np.abs(y[i]) < 1e-4
        num_zeros = np.sum(mask)
        filterd_y = y[i][~mask]
        hist, _ = np.histogram(y[i], bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2  # Calculate bin centers
        line_color = plt.plot(bin_centers, hist, label=f"Channel {i}")[0].get_color()

        channel_max = np.max(filterd_y)


        q99 = np.percentile(valid_feat[i].cpu().detach(), 99.0, axis=0)
        q1 = np.percentile(valid_feat[i].cpu().detach(), 1.0, axis=0)


        plt.axvline(q1, color=line_color, linestyle='-', linewidth=0.8)
        plt.axvline(q99, color=line_color, linestyle='-', linewidth=0.8)
        plt.axvline(channel_max, color=line_color, linestyle='--', linewidth=0.8)

        median = np.median(y[i], axis=0)
        plt.axvline(median, color=line_color, linestyle='-', linewidth=0.8)

    plt.show()

class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            # if idx == 0:
            #     Norm = SparseBatchNorm2d
            # else:
            #     Norm = nn.BatchNorm2d
            Norm = nn.BatchNorm2d
            cur_layers = [
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=1, bias=False
                ),
                Norm(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU6()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    Norm(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU6()
                ])

            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or (stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)):
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int32)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

        self.DEBUG = False

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features

        masks = data_dict["masks"]
        for i in range(len(self.blocks)):
            # x = self.blocks[i](x)
            block = self.blocks[i]
            for layer in block:
                if type(layer).__name__ in ["SparseBatchNorm2d", "BatchNorm2d", "InstanceNorm2d", "QuantBatchNorm2d", "ReLU6", "QuantConv2d"]:
                    if self.DEBUG:
                        print(i, "------------", layer)
                        bev_feat_reshape = x.permute(0, 2, 3, 1)
                        bev_feat_reshape = x.reshape(-1, x.size(1))
                        org_mean = bev_feat_reshape.mean(dim=0)
                        org_var = bev_feat_reshape.var(dim=0)
                        print("org_mean", torch.max(org_mean), torch.min(org_mean))
                        print("org_var", torch.max(org_var), torch.min(org_var))
                        print("input=",torch.max(x), torch.min(x))
                        if i ==0:
                            visualize_activation_histogram(x, masks, type(layer).__name__ + str(i) + "_input")
                    if type(layer).__name__ == "SparseBatchNorm2d":
                        # if i == 0:
                        #     print(i, "------------", layer)
                        #     bev_feat_reshape = x.permute(0, 2, 3, 1)
                        #     bev_feat_reshape = x.reshape(-1, x.size(1))
                        #     org_mean = bev_feat_reshape.mean(dim=0)
                        #     org_var = bev_feat_reshape.var(dim=0)
                        #     print("org_mean", torch.max(org_mean), torch.min(org_mean))
                        #     print("org_var", torch.max(org_var), torch.min(org_var))
                        #     print("input=",torch.max(x), torch.min(x))
                        #     visualize_activation_histogram(x, type(layer).__name__ + str(i) + "_input")
                        x = layer(x, masks)
                        # if i == 0:
                        #     bev_feat_reshape = x.permute(0, 2, 3, 1)
                        #     bev_feat_reshape = x.reshape(-1, x.size(1))
                        #     norm_mean = bev_feat_reshape.mean(dim=0)
                        #     norm_var = bev_feat_reshape.var(dim=0)
                        #     print("norm_mean", torch.max(norm_mean), torch.min(norm_mean))
                        #     print("norm_var", torch.max(norm_var), torch.min(norm_var))
                        #     print("output=",torch.max(x), torch.min(x))
                        #     visualize_activation_histogram(x, type(layer).__name__ + str(i) + "_output")
                    # elif type(layer).__name__ == "QuantBatchNorm2d" and i == 0:
                    #     # print(layer)
                    #     # print(layer.bias)
                    #     # print(layer.running_mean)
                    #     layer.bias *= 0
                    #     layer.running_mean *= 0
                    #     # exit()
                    #     x = layer(x)
                    else:
                        x = layer(x)
                    if self.DEBUG:
                        bev_feat_reshape = x.permute(0, 2, 3, 1)
                        bev_feat_reshape = x.reshape(-1, x.size(1))
                        norm_mean = bev_feat_reshape.mean(dim=0)
                        norm_var = bev_feat_reshape.var(dim=0)
                        print("norm_mean", torch.max(norm_mean), torch.min(norm_mean))
                        print("norm_var", torch.max(norm_var), torch.min(norm_var))
                        print("output=",torch.max(x), torch.min(x))
                        if i == 0:
                            visualize_activation_histogram(x, masks, type(layer).__name__ + str(i) + "_output")   
                else:
                    x = layer(x)
                
            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict

class BaseBEVBackboneV1(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2

        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)

        num_levels = len(layer_nums)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.Conv2d(
                    num_filters[idx], num_filters[idx], kernel_size=3,
                    stride=1, padding=1, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['multi_scale_2d_features']

        x_conv4 = spatial_features['x_conv4']
        x_conv5 = spatial_features['x_conv5']

        ups = [self.deblocks[0](x_conv4)]

        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[1](x))

        x = torch.cat(ups, dim=1)
        x = self.blocks[0](x)

        data_dict['spatial_features_2d'] = x

        return data_dict


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        padding: int = 1,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample_layer(x)

        out += identity
        out = self.relu2(out)

        return out


class BaseBEVResBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                BasicBlock(c_in_list[idx], num_filters[idx], layer_strides[idx], 1, True)
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    BasicBlock(num_filters[idx], num_filters[idx])
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters) if len(num_upsample_filters) > 0 else sum(num_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict

class SparseBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(SparseBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        assert track_running_stats == True, "track_running_stats=False is not supported"

        self.USE_BIAS = True

        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            if self.USE_BIAS:
                self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

        self.USE_MEAN = True

    def forward(self, feat, masks):
        if self.training:
            kernel_size_h = masks.size(2) / feat.size(2)
            kernel_size_w = masks.size(3) / feat.size(3)


            feat_permute = feat.permute(0, 2, 3, 1)

            if self.USE_MEAN:
                # mean = feat.mean(dim=[0, 2, 3])
                mean = torch.median(feat_permute.reshape(-1, feat.size(1)), dim=0).values



            assert kernel_size_h % 2 == 0, "kernel_size_h="+str(kernel_size_h)+"is not supported right now"
            assert kernel_size_w % 2 == 0, "kernel_size_w"+str(kernel_size_w)+"is not supported right now"

            masks = torch.nn.functional.max_pool2d(masks.float(), kernel_size=(int(kernel_size_h),int(kernel_size_w))).bool()

            mask_expanded = masks.expand(-1, feat.size(1), -1, -1)

            mask_expanded = mask_expanded.permute(0, 2, 3, 1)
            feat_permute = feat.permute(0, 2, 3, 1)

            valid_feat = feat_permute[mask_expanded]
            valid_feat = valid_feat.reshape(-1, feat.size(1))

            if self.track_running_stats:
                with torch.no_grad():                    
                    unbiased_var = torch.var(valid_feat, dim=[0], correction=1)
                    if self.USE_MEAN:
                        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbiased_var
                    self.num_batches_tracked += 1

            biased_var = torch.var(valid_feat, dim=[0], correction=0)
            q75 = torch.quantile(valid_feat.float(), 0.90, dim=0)
            q25 = torch.quantile(valid_feat.float(), 0.10, dim=0)
            biased_var = q75 - q25
            
            print(torch.max(biased_var), torch.min(biased_var))
            # print(torch.max(biased_var_quant), torch.min(biased_var_quant))
            if self.USE_MEAN:
                feat_normalized = (feat - mean.view(1, self.num_features, 1, 1)) / torch.sqrt(biased_var.view(1, self.num_features, 1, 1) + self.eps)
            else:
                feat_normalized = feat / torch.sqrt(biased_var.view(1, self.num_features, 1, 1) + self.eps)

        else:
            if self.track_running_stats:
                if self.USE_MEAN:
                    feat_normalized = (feat - self.running_mean.view(1, self.num_features, 1, 1)) / torch.sqrt(self.running_var.view(1, self.num_features, 1, 1) + self.eps)
                else:
                    feat_normalized = feat / torch.sqrt(self.running_var.view(1, self.num_features, 1, 1) + self.eps)

            else:
                raise ValueError("running_stats must be True during inference")

        if self.affine:
            if self.USE_BIAS:
                return feat_normalized * self.weight.view(1, self.num_features, 1, 1) + self.bias.view(1, self.num_features, 1, 1)
            else:
                return feat_normalized * self.weight.view(1, self.num_features, 1, 1)
        else:
            return feat_normalized

