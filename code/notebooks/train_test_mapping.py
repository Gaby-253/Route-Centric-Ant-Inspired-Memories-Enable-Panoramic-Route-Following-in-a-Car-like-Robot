# File: create_dataset_mapping
# Author: Gabriel Gattaux
# Date: 2024-03-05

import argparse
import antcar_sim
import os
from datetime import datetime
import subprocess
import numpy as np
import sys

def get_args():
    parser = argparse.ArgumentParser(description="This simulator represent the visual environement of a scene with several views in a familiarity-like manner")
    parser.add_argument('-irl',"--inputsroutelearn", required=True, help="Input learning path where image are stoked",type=str)
    parser.add_argument('-it',"--inputtest", required=True, help="Input tested path where image are stoked",type=str)
    parser.add_argument('-o',"--output", required=True, help="Input tested path where image are stoked")
    parser.add_argument('-p', '--paramsnet', required=True, help='Mushroom Bodies BNN parameters, the length of the list specify the mb_nb specifie as KC_nb,pntokc_syn_nb,seed,kc_norm,KC_nb2,...', type=str)
    parser.add_argument("-osl","--oscillearn", required=True, help="The 'in silico' rotation during learning and step", type=str)
    parser.add_argument("-vis","--paramsvision", required=True, help="The parameters for postprocessing vision : resolution,gaussian sigma",type=str)
    parser.add_argument('-ipl',"--inputsplacelearn", help="Input learning places where image are stoked",type=str)
    parser.add_argument("-sav","--save", required=True, help="The flag is wether the image generated during encoding process are saved or not on the disk",type=bool)

    return parser.parse_args()

def get_github_version():
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        return git_hash
    except subprocess.CalledProcessError:
        return 'Unknown'

def write_params_to_file(args, github_version):
    current_time = datetime.now().strftime('%d%m%Y%H%M')
    out_folder = args.output
    print(out_folder)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    with open(out_folder + '/launch_details.txt', 'w') as file:
        file.write('Parameters:\n')
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")
        file.write('\nGitHub Version: ')
        file.write(github_version)
        file.write(f'\n\nTimestamp: {current_time}')

def cleanup_and_exit(ANTSIM):
    # Cleanup operations
    del ANTSIM
    sys.exit(0)

def main():
    args = get_args()
    github_version = get_github_version()
    write_params_to_file(args, github_version)

    params_learn = np.array([float(item) for item in args.oscillearn.split(',')])
    # params_sim = np.array([int(item) for item in args.paramssim.split(',')])

    src_learning = args.inputsroutelearn
    inputs_test_src = args.inputtest

    max_oscil = params_learn[0]
    step_oscil = params_learn[1]

    try:
        ANTSIM = antcar_sim.AntCarSim(args)

        src_learn_mb1 = os.path.join(args.output, "images/learning1/augmented/")
        src_learn_mb2 = os.path.join(args.output, "images/learning2/augmented/")

        arr_0_to_45 = np.arange(0,max_oscil + step_oscil,step_oscil,dtype="int")
        arr_0_to_m45 = np.arange(0,-(max_oscil + step_oscil),-step_oscil,dtype="int")

        augmentation_oscillation_train = np.array([arr_0_to_45,arr_0_to_m45])
        augmentation_oscillation_test = np.arange(-180,180+step_oscil,step_oscil,dtype="int")

        src_test = os.path.join(args.output, "images/test/augmented/")

        _,traj_route_train = ANTSIM.read_raw_img_folder(src_learning)

        ANTSIM.save_csv(traj_route_train,'traj_train.csv')

        ANTSIM.augment_dataset(src_learning, src_learn_mb1, augmentation_oscillation_train[0], batch_size=20)
        ANTSIM.augment_dataset(src_learning, src_learn_mb2, augmentation_oscillation_train[1], batch_size=20)
        ANTSIM.augment_dataset(inputs_test_src, src_test, augmentation_oscillation_test, batch_size=20)

        mapping_mb1, mbs1 = ANTSIM.train(0, src_learn_mb1,batch_size=20)
        mapping_mb2, mbs2 = ANTSIM.train(1, src_learn_mb2,batch_size=20)

        mapping_test = ANTSIM.test(src_test,traj_route_train,batch_size=40)

        compressed_memory_list = np.vstack((mbs1, mbs2)).T
        mapping_learning = np.concatenate((mapping_mb1, mapping_mb2), axis=1)


        ANTSIM.save_csv(mapping_learning,'mapping_learning.csv')
        ANTSIM.save_csv(compressed_memory_list,'memories_learning.csv')
        ANTSIM.save_csv(mapping_test,'mapping_test.csv')
        
        cleanup_and_exit(ANTSIM)
        

    except KeyboardInterrupt:
        print("KeyboardInterrupt: Cleaning up and exiting...")
        cleanup_and_exit(ANTSIM)

if __name__ == "__main__":
    main()
