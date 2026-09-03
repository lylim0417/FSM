# Copyright 2023 Toyota Research Institute. All rights reserved.
#
# Fixed-calibration training baseline derived from the released VIDAR
# BaseFSM / SelfCalibFSM implementation.
#
# Research reconstruction:
#   - uses known camera intrinsics
#   - uses known rigid multi-camera extrinsics
#   - learns temporal ego-motion with PoseNet
#   - uses temporal + spatial photometric self-supervision
#   - does NOT use GT depth
#   - does NOT use GT ego-motion scale injection
#   - does NOT learn camera extrinsics

from vidar.arch.models.releaseSesc.BaseFSM import BaseFSM
from vidar.arch.networks.layers.selffsm.dataset_interface_method import (
    get_relative_poses_from_base_cam,
    get_unbroken_extrinsic,
    run_break_if_not_yet,
)
from vidar.geometry.camera import Camera
from vidar.utils.config import cfg_has


class FixedCalibFSM(BaseFSM):
    """Fixed-calibration self-supervised multi-camera depth baseline."""

    def __init__(self, cfg):
        super().__init__(cfg)

        self.freeze_posenet = cfg_has(cfg.model, 'freeze_posenet', False)
        self.freeze_depthnet = cfg_has(cfg.model, 'freeze_depthnet', False)
        self.mono_coeff = cfg_has(cfg.model, 'mono_loss_coeff', 1.0)

        # Guardrails for the camera-only self-supervised baseline.
        if self.gt_to_scale_injection:
            raise ValueError(
                'FixedCalibFSM requires gt_to_scale_injection: False'
            )

        if self.use_gt_pose:
            raise ValueError(
                'FixedCalibFSM requires use_gt_pose: False'
            )

    def forward(self, batch, epoch=0):
        """Forward pass for training or evaluation."""

        # --------------------------------------------------------------
        # Evaluation
        # --------------------------------------------------------------
        if not self.training:
            intrinsics = batch['intrinsics'][self.tgt_key]

            depths = self.get_mul_res_depth(
                rgb=batch['rgb'][self.tgt_key],
                intrinsics=intrinsics,
            )

            return {
                'predictions': {
                    'depth': {
                        self.tgt_key: [depth for depth in depths]
                    }
                }
            }

        # --------------------------------------------------------------
        # Training
        # --------------------------------------------------------------
        self.set_flip_status(batch_arg=batch)

        out = self.forward_fixedcalib(batch)

        broken_rgb = out['rgb']
        broken_intrinsics = out['intrinsics']
        broken_depth = out['pred_depth']
        pred_from_main_cam = out['pred_from_main_cam']

        broken_keys = list(broken_rgb.keys())

        # Each Camera receives:
        #   Twc = T_camera_from_canonical_camera
        #
        # For current cameras:
        #   T_Ci_from_C0 comes from fixed metric calibration.
        #
        # For temporal contexts:
        #   PoseNet motion is composed with the fixed camera extrinsics.
        cameras = {
            key: Camera(
                broken_intrinsics[(0, key[1])],
                broken_rgb[key],
                pred_from_main_cam[key],
            )
            for key in broken_keys
        }

        losses = []

        # --------------------------------------------------------------
        # Temporal self-supervision:
        # same physical camera at t-1 / t+1
        # --------------------------------------------------------------
        if self.mono_coeff > 0.0:
            mono_loss = self.get_photometric_loss(
                broken_keys=broken_keys,
                valid_mask_type='if_ddad',
                ctx_generator=self.get_mono_pair,
                broken_depth=broken_depth,
                broken_rgb=broken_rgb,
                broken_cameras=cameras,
                with_smoothness=True,
            )

            # Deliberate reconstruction correction:
            # released SelfCalibFSM contains
            #     self.mono_coeff + mono_loss
            # Here mono_coeff acts as an actual multiplicative weight.
            losses.append(self.mono_coeff * mono_loss)

        # --------------------------------------------------------------
        # Spatial / spatio-temporal multi-camera self-supervision
        # --------------------------------------------------------------
        if self.stereo_coeff > 0.0:
            stereo_loss = self.get_photometric_loss(
                broken_keys=broken_keys,
                valid_mask_type='sky_ground',
                ctx_generator=self.get_valid_stereo_pair,
                broken_depth=broken_depth,
                broken_rgb=broken_rgb,
                broken_cameras=cameras,
                with_smoothness=False,
                use_default_automask=False,
            )

            losses.append(self.stereo_coeff * stereo_loss)

        if len(losses) == 0:
            raise RuntimeError(
                'FixedCalibFSM has no active self-supervised losses.'
            )

        return {
            'metrics': {},
            'loss': sum(losses),
        }

    def forward_fixedcalib(self, batch):
        """
        Generate depth, temporal pose and fixed-calibration camera transforms.

        Important transform convention used here:

            T_destination_from_source

        get_relative_poses_from_base_cam() produces:

            T_Ci_from_C0

        where camera index 0 is the canonical camera of the six-camera batch.
        """

        rgb = batch['rgb']
        intrinsics = batch['intrinsics']
        pose_gt = batch['pose']

        # --------------------------------------------------------------
        # Fixed rigid multi-camera calibration
        #
        # DDAD's released SESC data path does not provide a separate
        # batch['extrinsics'] field. In that case, current-frame pose
        # entries encode the calibration structure expected by BaseFSM:
        #
        #   camera index 0 -> canonical/global pose entry
        #   camera indices 1...N -> camera poses relative to canonical
        #
        # Follow the released SelfCalibFSM convention exactly.
        # --------------------------------------------------------------
        if 'extrinsics' in batch:
            if self.broken:
                bxcamx4x4 = get_unbroken_extrinsic(
                    batch['extrinsics']
                )
            else:
                bxcamx4x4 = batch['extrinsics'][0]

            fixed_extrinsics = get_relative_poses_from_base_cam(
                bxcamx4x4
            )

        else:
            if self.broken:
                unbroken_pose = {
                    0: get_unbroken_extrinsic(batch['pose'])
                }
            else:
                unbroken_pose = batch['pose']

            fixed_extrinsics = self.pose2extrinsics_from_maincam(
                unbroken_pose
            )


        # --------------------------------------------------------------
        # Learned temporal ego-motion
        #
        # Explicitly pass NO GT pose-scale information.
        # --------------------------------------------------------------
        predicted_temporal_pose = self.get_broken_posenet(
            rgb,
            pose_freeze=self.freeze_posenet,
            gt_to_scale_injection=None,
        )

        # Compose learned temporal motion with the known rigid
        # camera calibration using the same machinery as SelfCalibFSM.
        pred_from_main_cam = self.get_pred_all_extrinsics(
            broken_posenet_out=predicted_temporal_pose,
            extrinsics=fixed_extrinsics,
        )

        # --------------------------------------------------------------
        # Convert input dictionaries into explicit (time, camera) keys
        # --------------------------------------------------------------
        broken_rgb = run_break_if_not_yet(rgb, 4)
        broken_intrinsics = run_break_if_not_yet(intrinsics, 3)

        # --------------------------------------------------------------
        # Shared depth network over all current cameras
        # --------------------------------------------------------------
        pred_depth = self.get_mul_depth_from_broken(
            broken_rgb=broken_rgb,
            broken_intrinsics=broken_intrinsics,
            freeze_depth=self.freeze_depthnet,
        )

        return {
            'rgb': broken_rgb,
            'intrinsics': broken_intrinsics,
            'pred_depth': pred_depth,
            'predicted_temporal_pose': predicted_temporal_pose,
            'fixed_extrinsics': fixed_extrinsics,
            'pred_from_main_cam': pred_from_main_cam,
        }
