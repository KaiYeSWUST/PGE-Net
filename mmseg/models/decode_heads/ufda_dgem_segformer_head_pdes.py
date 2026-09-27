import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmcv.cnn import ConvModule


class PDEOperator(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def gradient(x):
        x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
        grad_x = (
            x_pad[:, :, 1:-1, 2:]
            - x_pad[:, :, 1:-1, :-2]
        ) * 0.5
        grad_y = (
            x_pad[:, :, 2:, 1:-1]
            - x_pad[:, :, :-2, 1:-1]
        ) * 0.5
        return grad_x, grad_y

    @staticmethod
    def divergence(flux_x, flux_y):
        flux_x_pad = F.pad(flux_x, (1, 1, 0, 0), mode='replicate')
        flux_y_pad = F.pad(flux_y, (0, 0, 1, 1), mode='replicate')
        div_x = (
            flux_x_pad[:, :, :, 2:]
            - flux_x_pad[:, :, :, :-2]
        ) * 0.5
        div_y = (
            flux_y_pad[:, :, 2:, :]
            - flux_y_pad[:, :, :-2, :]
        ) * 0.5
        return div_x + div_y

    def forward(self, x):
        return self.gradient(x)

class DGEM(nn.Module):
    def __init__(self, in_channels, stage1_channels, num_classes=2):
        super(DGEM, self).__init__()
        self.num_classes = num_classes
        self.intercepted_vars = {}
        self.pde_operator = PDEOperator()
        self.phys_scale = nn.Parameter(torch.ones(1) * 1.2)

    def forward(
            self,
            joint_mask,
            aux_logits,
            nfim01_feat,
            f_boundary,
            uncertainty_mask,
            global_feat,
            raw_img,
    ):
        B_val, C_feat, Hb, Wb = f_boundary.shape
        _, C_raw, H_raw, W_raw = raw_img.shape

        if C_raw >= 4:
            raw_red = raw_img[:, 0:1, :, :]
            raw_green = raw_img[:, 1:2, :, :]
            raw_blue = raw_img[:, 2:3, :, :]
            raw_nir = raw_img[:, 3:4, :, :]
        else:
            raw_red = None
            raw_green = None
            raw_nir = raw_img[:, 0:1, :, :]
        nir_flat = raw_nir.reshape(B_val, 1, -1)
        nir_min = nir_flat.amin(dim=2, keepdim=True).view(B_val, 1, 1, 1)
        nir_max = nir_flat.amax(dim=2, keepdim=True).view(B_val, 1, 1, 1)
        nir_range = nir_max - nir_min
        nir_norm = (raw_nir - nir_min) / nir_range.clamp_min(1e-6)
        nir_norm = torch.where(
            nir_range > 1e-6,
            nir_norm,
            torch.zeros_like(nir_norm)
        )

        if not self.training:
            self.intercepted_vars['checking_nir_base'] = nir_norm.detach().clone()
        if raw_green is not None:
            r = torch.sigmoid(raw_red)
            g = torch.sigmoid(raw_green)
            b = torch.sigmoid(raw_blue)
            n = torch.sigmoid(raw_nir)
            ndwi = (g - n) / (g + n + 0.05)
            albedo = (r + g + b) / 3.0
            dynamic_thresh = 0.35 - torch.clamp(albedo * 1.5, max=0.27)
            physics_energy = torch.clamp(ndwi - dynamic_thresh, min=0.0) * 3.0
            physics_energy_pad = F.pad(physics_energy, (7, 7, 7, 7), mode='replicate')
            local_support = F.avg_pool2d(
                physics_energy_pad,
                kernel_size=15,
                stride=1,
                padding=0
            )
            is_lake_pattern = (local_support > 0.15).to(physics_energy.dtype)
            amplified_energy = physics_energy + is_lake_pattern * physics_energy * 2.0
            m_raw = torch.clamp(amplified_energy * 2.0, min=0.0, max=1.0)
            m_raw_pad = F.pad(m_raw, (2, 2, 2, 2), mode='replicate')
            local_density = F.avg_pool2d(
                m_raw_pad,
                kernel_size=5,
                stride=1,
                padding=0
            )
            g_den = (local_density > 0.25).to(m_raw.dtype)
            m_blockade = m_raw * g_den
            stifle_penalty = (1.0 - m_blockade) * 2.5

            if not self.training:
                self.intercepted_vars['checking_ndwi'] = ndwi.detach().clone()
                self.intercepted_vars['checking_dynamic_thresh'] = dynamic_thresh.detach().clone()
                self.intercepted_vars['checking_m_blockade'] = m_blockade.detach().clone()
                self.intercepted_vars['checking_stifle_penalty'] = stifle_penalty.detach().clone()
        else:
            stifle_penalty = torch.zeros_like(nir_norm)
        phi_init_highres = F.interpolate(f_boundary, size=(H_raw, W_raw), mode='bilinear', align_corners=False)
        water_prob_init = (
            F.softmax(phi_init_highres.detach(), dim=1)[:, 1:2, :, :]
            if C_feat == 2
            else torch.sigmoid(phi_init_highres.detach()[:, 0:1, :, :])
        )
        anchor_mask = (water_prob_init > 0.8).to(nir_norm.dtype)
        b_mu, b_sigma = [], []
        for b in range(B_val):
            mask_b = (anchor_mask[b, 0] == 1)
            if mask_b.sum() > 50:
                vals = nir_norm[b, 0][mask_b]
                b_mu.append(vals.mean().detach())
                b_sigma.append(vals.std(unbiased=False).detach().clamp_min(0.02))
            else:
                b_mu.append(torch.tensor(0.2, device=nir_norm.device))
                b_sigma.append(torch.tensor(0.1, device=nir_norm.device))
        mu_water = torch.stack(b_mu).view(B_val, 1, 1, 1)
        sigma_water = torch.stack(b_sigma).view(B_val, 1, 1, 1)
        dx_n, dy_n = self.pde_operator(nir_norm)
        nir_grad = torch.sqrt(dx_n ** 2 + dy_n ** 2 + 1e-8)
        physical_barrier = torch.exp(-15.0 * nir_grad)
        dist_sq = torch.clamp(nir_norm - mu_water, min=0.0) ** 2
        gauss_attraction = torch.exp(-dist_sq / (2 * sigma_water ** 2))
        base_force = 2.0 * gauss_attraction - 1.0
        final_balloon_force = base_force * physical_barrier * self.phys_scale
        polarized_phys = torch.tanh(final_balloon_force * 2.5)
        z_score_bright = F.relu(nir_norm - mu_water) / (sigma_water + 1e-8)
        z_score_dark = F.relu(mu_water - nir_norm) / (sigma_water + 1e-8)
        bright_penalty = 2.0 * F.relu(z_score_bright - 1.5)
        dark_penalty = 1.0 * F.relu(z_score_dark - 2.5)
        z_score_penalty = bright_penalty + dark_penalty
        total_spectral_penalty = stifle_penalty + z_score_penalty
        sharpened_phys = torch.clamp(
            polarized_phys - total_spectral_penalty,
            min=-2.5,
            max=1.0
        )
        aux_probs = F.softmax(aux_logits, dim=1)
        if self.num_classes == 2:
            water_prob = aux_probs[:, 1:2, :, :]
        else:
            water_prob = aux_probs[:, 0:1, :, :]

        water_prob_hr = F.interpolate(
            water_prob,
            size=(H_raw, W_raw),
            mode='bilinear',
            align_corners=False
        )
        water_prob_hr = torch.clamp(water_prob_hr, 0.0, 1.0)
        m_sem_w = (water_prob_hr >= 0.5).to(sharpened_phys.dtype)
        m_sem_b = (water_prob_hr < 0.5).to(sharpened_phys.dtype)
        m_phy_w = (sharpened_phys > 0.0).to(sharpened_phys.dtype)
        m_phy_b = (sharpened_phys < 0.0).to(sharpened_phys.dtype)
        joint_mask_hr = F.interpolate(
            joint_mask,
            size=(H_raw, W_raw),
            mode='bilinear',
            align_corners=False
        )
        joint_mask_hr = torch.clamp(joint_mask_hr, 0.0, 1.0)
        m_u = (joint_mask_hr > 0.5).to(sharpened_phys.dtype)
        m_c = 1.0 - m_u
        m_fp = m_c * m_sem_w * m_phy_b
        m_fn = m_c * m_sem_b * m_phy_w
        m_conf_hard = torch.clamp(m_fp + m_fn, min=0.0, max=1.0)
        physical_direction_valid = torch.clamp(
            m_phy_w + m_phy_b,
            min=0.0,
            max=1.0
        )
        m_conf_pot = m_u * physical_direction_valid
        m_conf_0 = torch.clamp(
            m_conf_hard + m_conf_pot,
            min=0.0,
            max=1.0
        )
        m_conf_pad = F.pad(m_conf_0, (1, 1, 1, 1), mode='replicate')
        n_conf = 9.0 * F.avg_pool2d(
            m_conf_pad,
            kernel_size=3,
            stride=1,
            padding=0
        )
        m_conf_r = m_conf_0 * (n_conf >= 2.0).to(m_conf_0.dtype)
        m_rez = F.max_pool2d(
            m_conf_r,
            kernel_size=7,
            stride=1,
            padding=3
        )
        if not self.training:
            self.intercepted_vars['gravity_field'] = final_balloon_force.detach().clone()
            self.intercepted_vars['corrected_gravity_field'] = sharpened_phys.detach().clone()
            self.intercepted_vars['water_probability'] = water_prob_hr.detach().clone()
            self.intercepted_vars['joint_mask_highres'] = joint_mask_hr.detach().clone()
            self.intercepted_vars['semantic_water_mask'] = m_sem_w.detach().clone()
            self.intercepted_vars['semantic_background_mask'] = m_sem_b.detach().clone()
            self.intercepted_vars['physical_water_mask'] = m_phy_w.detach().clone()
            self.intercepted_vars['physical_background_mask'] = m_phy_b.detach().clone()
            self.intercepted_vars['high_uncertainty_mask'] = m_u.detach().clone()
            self.intercepted_vars['explicit_conflict_mask'] = m_conf_hard.detach().clone()
            self.intercepted_vars['potential_conflict_mask'] = m_conf_pot.detach().clone()
            self.intercepted_vars['initial_conflict_mask'] = m_conf_0.detach().clone()
            self.intercepted_vars['reliable_conflict_mask'] = m_conf_r.detach().clone()
            self.intercepted_vars['restricted_evolution_zone'] = m_rez.detach().clone()

        return m_rez, sharpened_phys

@MODELS.register_module()
class UFDADGEMSegformerHeadPDES(BaseDecodeHead):
    def __init__(
            self,
            interpolate_mode='bilinear',
            aux_loss_weight=0.4,
            pde_steps=5,
            pde_dt=0.1,
            pde_alpha=0.2,
            pde_beta=0.5,
            pde_phi=0.5,
            **kwargs
    ):
        kwargs.pop('aux_loss_weight', None)
        super().__init__(input_transform='multiple_select', **kwargs)
        self.interpolate_mode = interpolate_mode
        self.pde_steps = int(pde_steps)
        self.pde_dt = float(pde_dt)
        self.pde_alpha = float(pde_alpha)
        self.pde_beta = float(pde_beta)
        self.pde_phi = float(pde_phi)

        if self.pde_steps < 1:
            raise ValueError('pde_steps must be greater than or equal to 1')
        if self.pde_dt <= 0.0:
            raise ValueError('pde_dt must be greater than 0')
        if self.pde_phi <= 0.0:
            raise ValueError('pde_phi must be greater than 0')

        self.pde_operator = PDEOperator()
        self.aux_loss_weight = aux_loss_weight
        num_inputs = len(self.in_channels)
        self.param_free_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels[i], self.channels, 1, bias=False),
                nn.SyncBatchNorm(self.channels),
                nn.ReLU(inplace=True)
            ) for i in range(num_inputs)
        ])
        self.fusion_conv = ConvModule(self.channels * num_inputs, self.channels, kernel_size=1, norm_cfg=self.norm_cfg)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(self.channels, self.channels // 2, kernel_size=3, padding=1, bias=False),
            nn.SyncBatchNorm(self.channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels // 2, self.num_classes, kernel_size=1)
        )
        self.body_enhance = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, groups=self.channels, bias=False),
            nn.SyncBatchNorm(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False),
            nn.SyncBatchNorm(self.channels)
        )
        self.boundary_conv = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        self.dgem_module = DGEM(
            in_channels=self.channels, stage1_channels=self.in_channels[0], num_classes=self.num_classes
        )
        self.re_fusion = ConvModule(
            self.channels + self.num_classes, self.channels, kernel_size=3, padding=1, norm_cfg=self.norm_cfg
        )
        self.cls_seg = nn.Sequential(
            nn.Conv2d(self.channels, self.channels // 2, kernel_size=3, padding=1, bias=False),
            nn.SyncBatchNorm(self.channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels // 2, self.num_classes, kernel_size=1)
        )
        self.intercepted_vars = {}

    def evolve_logits_with_pde(self, logits, f_sharp, m_rez):
        if logits.shape[1] != 2:
            raise ValueError(
                'The current PDE evolution is defined by the log-odds of the water body/background binary classification.'
                f'But we received {logits.shape[1]}channels of categories'
            )
        if f_sharp.shape[2:] != logits.shape[2:]:
            is_downsampling = (
                    f_sharp.shape[-2] > logits.shape[-2]
                    or f_sharp.shape[-1] > logits.shape[-1]
            )
            if is_downsampling:
                f_sharp = F.interpolate(
                    f_sharp,
                    size=logits.shape[2:],
                    mode='area'
                )
            else:
                f_sharp = F.interpolate(
                    f_sharp,
                    size=logits.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
        if m_rez.shape[2:] != logits.shape[2:]:
            m_rez = F.interpolate(
                m_rez,
                size=logits.shape[2:],
                mode='nearest'
            )
        f_sharp = f_sharp.to(dtype=logits.dtype)
        m_rez = m_rez.to(dtype=logits.dtype)
        grav_x, grav_y = self.pde_operator.gradient(f_sharp)
        grav_norm_sq = grav_x.square() + grav_y.square()
        diffusion_coeff = torch.exp(
            -grav_norm_sq / (self.pde_phi ** 2 + 1e-8)
        )
        phi_0 = logits[:, 1:2] - logits[:, 0:1]
        phi = phi_0
        for _ in range(self.pde_steps):
            phi_x, phi_y = self.pde_operator.gradient(phi)
            flux_x = diffusion_coeff * phi_x
            flux_y = diffusion_coeff * phi_y
            diffusion_term = self.pde_operator.divergence(flux_x, flux_y)
            advection_term = grav_x * phi_x + grav_y * phi_y
            rhs = (
                    self.pde_alpha * diffusion_term
                    - self.pde_beta * advection_term
            )
            phi = phi + self.pde_dt * rhs * m_rez
        delta_phi = phi - phi_0
        evolved_logits = torch.cat(
            [
                logits[:, 0:1] - 0.5 * delta_phi,
                logits[:, 1:2] + 0.5 * delta_phi
            ],
            dim=1
        )
        return evolved_logits, diffusion_coeff, grav_x, grav_y

    def forward(self, inputs):
        raw_img = inputs[-1]
        inputs_for_feat = inputs[:-1]
        nfim01_feat = inputs_for_feat[0]

        inputs_res = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs_res)):
            x = self.param_free_mlps[idx](inputs_res[idx])
            if idx != 0:
                x = F.interpolate(x, size=inputs_res[0].shape[2:], mode=self.interpolate_mode, align_corners=False)
            outs.append(x)
        global_feat = torch.cat(outs, dim=1)
        global_feat = self.fusion_conv(global_feat)
        aux_logits = self.uncertainty_head(global_feat)
        probs_soft = F.softmax(aux_logits, dim=1)
        probs_soft = torch.clamp(probs_soft, min=1e-8, max=1.0 - 1e-8)
        entropy = -torch.sum(probs_soft * torch.log2(probs_soft), dim=1, keepdim=True)
        entropy_mask = entropy / math.log2(self.num_classes)
        probs_raw = F.softmax(aux_logits, dim=1)
        prob_max = F.max_pool2d(probs_raw, kernel_size=3, padding=1, stride=1)
        prob_min = -F.max_pool2d(-probs_raw, kernel_size=3, padding=1, stride=1)
        spatial_mask = torch.max(prob_max - prob_min, dim=1, keepdim=True)[0]
        max_pool_nfim = F.max_pool2d(nfim01_feat, kernel_size=3, padding=1, stride=1)
        min_pool_nfim = -F.max_pool2d(-nfim01_feat, kernel_size=3, padding=1, stride=1)
        physical_grad = max_pool_nfim - min_pool_nfim
        physical_mask_raw = physical_grad.mean(dim=1, keepdim=True)
        if physical_mask_raw.shape[2:] != entropy_mask.shape[2:]:
            physical_mask_raw = F.interpolate(physical_mask_raw, size=entropy_mask.shape[2:], mode='bilinear',
                                              align_corners=False)
        physical_mask = torch.tanh(physical_mask_raw)
        semantic_gate = (entropy_mask > 0.01).float() * entropy_mask
        physical_mask = physical_mask * semantic_gate
        joint_mask = torch.max(torch.max(entropy_mask, spatial_mask), physical_mask)
        joint_mask = torch.clamp(joint_mask, 0.0, 1.0)
        joint_mask = torch.pow(joint_mask + 1e-7, 0.7)
        joint_mask = torch.clamp(joint_mask, 0.0, 1.0)

        if not self.training:
            self.intercepted_vars['ufda_entropy'] = entropy_mask.detach().clone()
            self.intercepted_vars['ufda_spatial'] = spatial_mask.detach().clone()
            self.intercepted_vars['ufda_physical'] = physical_mask.detach().clone()
            self.intercepted_vars['ufda_joint_base'] = joint_mask.detach().clone()

        f_body = global_feat * (1.0 - joint_mask * 0.5)
        f_body_out = self.body_enhance(f_body)
        f_boundary = self.boundary_conv(global_feat)
        m_rez, phys_field_final = self.dgem_module(
            joint_mask=joint_mask,
            aux_logits=aux_logits,
            nfim01_feat=nfim01_feat,
            f_boundary=f_boundary,
            uncertainty_mask=entropy_mask,
            global_feat=global_feat,
            raw_img=raw_img
        )

        if not self.training and hasattr(self.dgem_module, 'intercepted_vars'):
            self.intercepted_vars.update(self.dgem_module.intercepted_vars)
            self.intercepted_vars['restricted_evolution_zone'] = m_rez.detach().clone()

        if self.training:
            phys_attn = torch.sigmoid(
                F.interpolate(
                    phys_field_final,
                    size=f_boundary.shape[2:],
                    mode='area'
                )
            )
            f_boundary_out = f_boundary * phys_attn
            f_concat = torch.cat([f_body_out, f_boundary_out], dim=1)
            base_feat = self.re_fusion(f_concat) + global_feat
            final_feat = base_feat + base_feat * phys_attn
            seg_logits = self.cls_seg(final_feat)
            seg_logits, _, _, _ = self.evolve_logits_with_pde(
                logits=seg_logits,
                f_sharp=phys_field_final,
                m_rez=m_rez
            )
            return (seg_logits, aux_logits, f_boundary)
        else:
            phys_attn_val = torch.sigmoid(
                F.interpolate(
                    phys_field_final,
                    size=f_boundary.shape[2:],
                    mode='area'
                )
            )
            f_boundary_val = f_boundary * phys_attn_val
            f_concat_val = torch.cat([f_body_out, f_boundary_val], dim=1)
            final_feat_val = self.re_fusion(f_concat_val) + global_feat
            seg_logits = self.cls_seg(final_feat_val)
            seg_logits_hr = F.interpolate(
                seg_logits,
                size=phys_field_final.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            final_evolved_logits, diffusion_coeff, grav_x, grav_y = (
                self.evolve_logits_with_pde(
                    logits=seg_logits_hr,
                    f_sharp=phys_field_final,
                    m_rez=m_rez
                )
            )
            self.intercepted_vars['pde_initial_logits'] = seg_logits_hr.detach().clone()
            self.intercepted_vars['pde_diffusion_coeff'] = diffusion_coeff.detach().clone()
            self.intercepted_vars['pde_gravity_x'] = grav_x.detach().clone()
            self.intercepted_vars['pde_gravity_y'] = grav_y.detach().clone()
            self.intercepted_vars['pde_final_logits'] = final_evolved_logits.detach().clone()
            return final_evolved_logits

    def loss_by_feat(self, seg_logits, batch_data_samples):
        seg_pred, aux_pred, f_boundary = seg_logits
        losses = super().loss_by_feat(seg_pred, batch_data_samples)

        aux_losses = super().loss_by_feat(aux_pred, batch_data_samples)
        for k, v in aux_losses.items():
            losses[f'aux_{k}'] = v * self.aux_loss_weight

        bound_losses = super().loss_by_feat(f_boundary, batch_data_samples)
        for k, v in bound_losses.items():
            losses[f'bound_{k}'] = v * 0.4
        return losses
