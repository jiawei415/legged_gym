# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class MaxCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.31] # x,y,z [m]
        offset = 0.2
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # 'joint_FL1': 0.1,   # [rad]
            # 'joint_HL1': 0.1,   # [rad]
            # 'joint_FR1': -0.1 ,  # [rad]
            # 'joint_HR1': -0.1,   # [rad]

            # 'joint_FL2': -0.8,     # [rad]
            # 'joint_HL2': -1.,   # [rad]
            # 'joint_FR2': -0.8,     # [rad]
            # 'joint_HR2': -1.,   # [rad]

            # 'joint_FL3': 1.5,   # [rad]
            # 'joint_HL3': 1.5,    # [rad]
            # 'joint_FR3': 1.5,  # [rad]
            # 'joint_HR3': 1.5,    # [rad]

            'joint_FL1': 0.2,  # [rad]
            'joint_HL1': 0.2,  # [rad]
            'joint_FR1': -0.2,  # [rad]
            'joint_HR1': -0.2,  # [rad]

            'joint_FL2': -0.8 - offset,  # [rad]
            'joint_HL2': -1.0 - offset,  # [rad]
            'joint_FR2': -0.8 - offset,  # [rad]
            'joint_HR2': -1.0 - offset,  # [rad]

            'joint_FL3': 1.5 + offset,  # [rad]
            'joint_HL3': 1.5 + offset,  # [rad]
            'joint_FR3': 1.5 + offset,  # [rad]
            'joint_HR3': 1.5 + offset,  # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/max/max.urdf'
        name = "max"
        foot_name = "None"
        penalize_contacts_on = ["1", "2"]
        terminate_after_contacts_on = ["body"]
        flip_visual_attachments = False
        fix_base_link = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0

    class viewer:
        ref_env = 0
        # pos = [5, 5, 3]  # [m]
        # lookat = [0., 0., 0.]  # [m]

        pos = [25, 8, 3]  # [m]
        lookat = [20.0, 3.8, 0.0]  # [m]


class MaxCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'max'
