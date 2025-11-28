from math import ceil
import torch
import torch.nn as nn
from vc_models.models.vit import model_utils
from torch.nn import functional as F


class Vc1Wrapper(nn.Module):
    """
    Wrapper for the VC1 visual encoder. This will automatically download the model if it's not already.
    """

    def __init__(self, im_obs_space, model_id=None, vc1_config=None):
        super().__init__()
        assert vc1_config is not None, "Make sure you pass vc1_config to Vc1Wrapper."
        self.vc1_config = vc1_config
        if model_id is None:
            model_id = model_utils.VC1_BASE_NAME
        print(f"loading {model_id}.")
        (
            self.net,
            self.embd_size,
            self.model_transforms,
            model_info,
        ) = model_utils.load_model(model_id)
        self._image_obs_keys = [k for k in im_obs_space.spaces.keys() if (k != "depth" and 'semantic' not in k)]
        print(self._image_obs_keys)

        # Count total # of channels
        self._n_input_channels = sum(
            im_obs_space.spaces[k].shape[2] for k in self._image_obs_keys
        )
        if self.vc1_config.is_2d_output and self.vc1_config.avg_pool_size:
            self.postprocess = nn.AvgPool2d(self.vc1_config.avg_pool_size, ceil_mode=True) if self.vc1_config.avg_pool_size > 1 else nn.Identity()
            self.out_dim = int(ceil(14/self.vc1_config.avg_pool_size))
            self.embd_size = self.net.embed_dim[-1] * 2  # cls token + embed dim
        else:
            self.postprocess = nn.Identity()
            self.out_dim = 1
	
    @property
    def is_blind(self):
        return self._n_input_channels == 0

    @torch.autocast("cuda", enabled=False)
    def forward(self, obs):
        # Extract tensors that are shape [batch_size, img_width, img_height, img_channels]
        feats = []
        imgs = [v for k, v in obs.items() if k in self._image_obs_keys]
        for img in imgs:
            if img.shape[-1] != 3:
                img = torch.concat([img]*3, dim=-1)
                scale_factor = 1.0
            else:
                scale_factor = 255.0

            img = self.model_transforms(img.permute(0, 3, 1, 2).contiguous() / scale_factor)

            feats.append(self.net(img))

        if len(feats) == 2:
            # feats = (feats[0] + feats[1])/2
            feats = torch.concat(feats, dim=-1)
        else:
            feats = feats[0]

        # if is_2d_output and avg_pool_size > 1, feats will be of shape 20, 197, 768
        # we need to first separate the cls token which is the first token in the sequence (1 + 196)
        # and then transform it to the desired output shape so that postprocessing can be applied correctly on the spatial dims 14x14
        # then we want to include back the cls token by concatenating it to the output channelwise
        if self.vc1_config.is_2d_output and self.vc1_config.avg_pool_size > 1:

            # Reshape feats to (batch_size, 14, 14, embd_size)
            batch_size = feats.shape[0]
            num_tokens = feats.shape[1] - 1
            assert num_tokens == 196, "Expected 196 tokens for 14x14 grid, got {}".format(num_tokens)
            cls_token = feats[:, 0]
            feats = feats[:, 1:].reshape(batch_size, self.net.final_spatial, self.net.final_spatial, self.net.embed_dim[-1])
            # Apply avg pooling if needed
            feats = self.postprocess(feats.permute(0, 3, 1, 2)) # 20, 768, out_dim, out_dim
            # concat the cls token back so that it becomes (batch_size, self.net.embed_dim[-1]*2, out_dim, out_dim) = 20, 768*2, out_dim, out_dim
            cls_token = cls_token.unsqueeze(-1).unsqueeze(-1)  # Reshape to (batch_size, embd_size, 1, 1)
            cls_token = cls_token.expand( batch_size, self.net.embed_dim[-1], self.out_dim, self.out_dim)
            feats = torch.cat((cls_token, feats), dim=1)  # Concatenate along the channel dimension
            # Ensure the shape is correct
            assert feats.shape[1] == self.net.embed_dim[-1]*2, f"Expected {self.net.embed_dim[-1]*2} channels, got {feats.shape[1]}"
            # Now feats is of shape (batch_size, 2*net.embed_dim[-1], out_dim, out_dim)
            # Reshape back to (batch_size, embd_size * out_dim * out_dim)
            return feats.flatten(1)
        
        return self.postprocess(feats).flatten(1)

    @property
    def output_shape(self):
        return (self.out_dim * self.out_dim * self.embd_size * len(self._image_obs_keys),)

    @property
    def feats_size(self):
        return self.embd_size * len(self._image_obs_keys)
    
    def set_grad_checkpointing(self):
        return self.net.set_grad_checkpointing()
