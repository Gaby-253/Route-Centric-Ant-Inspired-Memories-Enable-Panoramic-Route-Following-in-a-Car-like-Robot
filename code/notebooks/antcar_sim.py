## Author: Gabriel Gattaux


import numpy as np
import memory
import cv2
from skimage import filters
import os
import imutils
import time
import pyproj
import pandas as pd 
from matplotlib.colors import Normalize
import matplotlib.cm as cm

class AntCarSim:
    """This Class provides a simulate agent that can learn from a viewbank and
    map another viewbank for visualization purposes"""

    def __init__(self,args):
        self._args = args
        self._inputs_train_route_src = getattr(args, 'inputsroutelearn', None)
        self._inputs_test_src = getattr(args, 'inputtest', None)
        
        inputsplacelearn = getattr(args, 'inputsplacelearn', None)
        self._inputs_train_places_src = inputsplacelearn.split(',') if inputsplacelearn else None
        
        paramsnet = getattr(args, 'paramsnet', "")
        self._params_net = np.reshape(np.array([float(item) for item in paramsnet.split(',')]), (-1, 4)) if paramsnet else None
        
        oscillearn = getattr(args, 'oscillearn', "")
        self._params_learn = np.array([float(item) for item in oscillearn.split(',')]) if oscillearn else None
        
        paramsvision = getattr(args, 'paramsvision', "")
        self._params_vision = np.array([int(item) for item in paramsvision.split(',')]) if paramsvision else None
        
        self._outputfolder = getattr(args, 'output', None)
        
        self._saving_falg = bool(getattr(args, 'save', False))
        # self._gps = bool(args.gps)

        self.augmented_idx = 0
        # self._inputs_train_route_src_augmented = [os.path.join(self._outputfolder, "images/learning0/augmented/"),os.path.join(self._outputfolder, "images/learning1/augmented/")]
        # self._inputs_test_src_augmented = os.path.join(self._outputfolder, "images/test/augmented/")

        self._mbs_nb = self._params_net.shape[0]
        self._mbs = []

        if self._mbs_nb > 1:
            for it in range(self._mbs_nb):
                self.add_mb(self._inputs_train_route_src, kc_nb=int(self._params_net[it,0]), kc_to_pn_syn=int(self._params_net[it,1]), kc_norm=self._params_net[it,3], seed=int(self._params_net[it,2]))
        else:
            self.add_mb(self._inputs_train_route_src, kc_nb=int(self._params_net[0]), kc_to_pn_syn=int(self._params_net[1]), kc_norm=self._params_net[3], seed=int(self._params_net[2]))
        
        self.__summary__()

    def save_csv(self,variable,outname):
            df = pd.DataFrame(variable)
            df.to_csv(os.path.join(self._outputfolder, outname), index=False)

    def add_mb(self, inputs_train_route_src, kc_nb=1000, kc_to_pn_syn=4, kc_norm=0.1, seed=None):
        """Add one MB to the list of MBs"""
        first_image_path = os.path.join(inputs_train_route_src, os.listdir(inputs_train_route_src)[0])
        first_image = cv2.imread(first_image_path)

        pn_nb = len(self.create_pn(first_image)[0])
        mb = memory.MushroomBody(
            PN_nb=pn_nb,
            KC_nb=kc_nb,
            KCtoPN_synapses=kc_to_pn_syn,
            KC_norm_param=kc_norm,
            seed=seed,
        )
        self._mbs.append(mb)

    def __summary__(self):
        print(self._mbs)

    def create_pn(self, image):
        """Function converting the raw image from the sensors
        to the corresponding Neural Projection"""
        resolution = self._params_vision[0]
        sigma = self._params_vision[1]
        image = image[:, :, 1]
        image = image.astype(np.uint8)
        image = filters.gaussian(image, sigma=sigma)
        img_resampled = cv2.resize(
            image, (int(resolution), int(resolution)), interpolation=cv2.INTER_NEAREST
        )
        # Sobel filter for edge detection
        img_sobel = filters.sobel(img_resampled)
        ra = resolution / 2
        cx = resolution / 2
        cy = resolution / 2
        # create a meshgrid of indices for the image
        x, y = np.meshgrid(np.arange(img_sobel.shape[0]), np.arange(img_sobel.shape[1]))
        # calculate istances from center of image
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        # create boolean masks for radii
        outer_mask = dist <= ra
        # apply masks to image
        pn = img_sobel[outer_mask]  # & inner_mask
        img_sobel = ((img_sobel - img_sobel.min()) * (1/(img_sobel.max() - img_sobel.min()) * 255)).astype('uint8')
        pn = ((pn - pn.min()) * (1/(pn.max() - pn.min()) * 255)).astype('uint8')
        return pn,img_sobel

    def latlon_to_xylocal(self,lat,lon):
        P = pyproj.Proj(proj='utm', zone=31, ellps='WGS84', preserve_units=True) # You can modify your zone
        XY=P(lat,lon)
        return float(XY[0]),float(XY[1])

    def sawtooth(self,theta):
        if theta > 180:
            theta -= 360
        elif theta < -180:
            theta += 360
        return theta

    def read_raw_img_folder(self,src,bool_all=True,start_idx=0,end_idx=0):
        """Read raw images from a folder, sort them and save them as a np array"""
        images = []
        trajs = []
        filenames = os.listdir(src)
        # sorted_filenames = sorted(filenames, key=lambda x: float(x.split("_")[0]))

        if bool_all:
            sorted_filenames = sorted(filenames, key=lambda x: float(x.split("_")[0]))
        else:
            sorted_filenames = sorted(filenames, key=lambda x: (float(x.split("_")[0]), float(x.split("_")[3].split(".")[0])))[start_idx:end_idx]

        for filename in sorted_filenames:
            path = os.path.join(src, filename)
            images.append(cv2.imread(path, cv2.IMREAD_COLOR))
            i, x, y, theta = filename.split("_", 4)
            theta = float(theta[:-4])  # Remove ".jpg" extension
            trajs.append([float(i), float(x), float(y), theta])
        return np.array(images), np.array(trajs)

    def rotating_imagesbank(self, images, in_silico_oscillation, trajs,place=0):
        """Categorises the images into different classes, here the classes are
        based on rotation"""
        augmented_images = []
        augmented_trajs = []
        for idx, img in enumerate(images):
            for angle in in_silico_oscillation:
                self.augmented_idx +=1
                if place == 1:
                    rot_ang = float(trajs[idx, 3]) - angle
                    current_angle = float(trajs[idx, 3]) + float(rot_ang)
                else:
                    rot_ang = angle
                    current_angle = float(trajs[idx, 3]) + float(rot_ang)

                current_angle = self.sawtooth(current_angle)
                augmented_images.append(imutils.rotate(img, rot_ang))
                augmented_trajs.append([self.augmented_idx, trajs[idx, 1], trajs[idx, 2], current_angle])

        return np.array(augmented_images), np.array(augmented_trajs)

    def augment_dataset(self, inputsrc, outputsrc, oscill, batch_size=10, place =0):
        """Augment the dataset by scanning in silico to have every rotation by desired angle and steps"""
        print(f"Data augmentation input: {inputsrc} \n output: {outputsrc}")
        filenames = os.listdir(inputsrc)
        num_images = len(filenames)
        num_batches = (num_images + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_images)
            # filenames = image_paths[start_idx:end_idx]
            # print(filenames)

            images_route, trajs = self.read_raw_img_folder(inputsrc,False,start_idx,end_idx)

            images_routes_rotated, trajs_rotated = self.rotating_imagesbank(images_route, oscill, trajs, place)

            self.save_images_processed(outputsrc,images_routes_rotated,trajs_rotated)

            print(f"Batch {batch_idx + 1}/{num_batches}.")

        self.augmented_idx = 0

    def create_pnbank(self, images):
        """Create pn bank from images bank"""
        pn = []
        img_sob = []
        for img in images:
            pn.append(self.create_pn(img)[0])
            img_sob.append(self.create_pn(img)[1])
        return np.array(pn), np.array(img_sob)

    def learn(self, pn_viewbank, mb):
        """Learning process on a given Memory"""
        unfamiliarities_learning = []
        kc_saturation = []
        sw_list = []
        kc_list = []
        pns_list = []

        sl = time.time()
        for _, pn in enumerate(pn_viewbank):
            unfamiliarities_learning.append(mb.get_unfamiliarity(pn))
            mb.refresh(pn)
            mb.learn()
            kc_saturation.append(np.mean(mb.W_KCtoMBON))
            pn = mb.pn_activity
            pns_list.append(pn.copy())
            sw = mb.W_KCtoMBON
            sw_list.append(sw.copy())
            kc_list.append(mb.kc_spikes)
        compressed_memory = mb.W_KCtoMBON
        el = time.time()
        # print(f"Learning in {el- sl:.2f}s")
        return (
            np.array(unfamiliarities_learning).reshape(-1, 1),
            np.array(kc_saturation).reshape(-1, 1),
            compressed_memory,
            np.array(sw_list),
            np.array(kc_list),
            np.array(pns_list),
        )

    def folder_with_most_files(self,src_image_learn):
        max_folder = None
        max_file_count = -1

        for folder in src_image_learn:
            file_count = len(os.listdir(folder))
            if file_count > max_file_count:
                max_file_count = file_count
                max_folder = folder

        return max_folder,max_file_count

    
    def train(self,mb_nb, src_image, batch_size=10):

        imgs_nb = len(os.listdir(src_image))

        unfamiliarities_learning_list = np.zeros((imgs_nb,1), dtype=float)
        kc_saturation = np.zeros((imgs_nb,1), dtype=float)
        trajs_learning_list = np.zeros((imgs_nb,4), dtype=float)

        # divide the trainning task into several batches
        num_batches = (imgs_nb + batch_size - 1) // batch_size

        subfolders = ["processed", "pns", "kcs","sw"]
        subfolder_paths = [os.path.join(self._outputfolder, f"images/learning{mb_nb+1}/{subfolder}/") for subfolder in subfolders]   
        
        sl = time.time()
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, imgs_nb)
            
            images_learning_batched, trajs_learning_batched = self.read_raw_img_folder(src_image,False,start_idx,end_idx)

            pns_learning_batched, image_process_learning_batched = self.create_pnbank(images_learning_batched)

            unfamiliarities_learning_, kc_sat, compressed_memory, sws, kcs, pns = self.learn(pns_learning_batched, self._mbs[mb_nb])

            unfamiliarities_learning_list[start_idx:end_idx] = unfamiliarities_learning_
            kc_saturation[start_idx:end_idx] = kc_sat
            trajs_learning_list[start_idx:end_idx,:] = trajs_learning_batched
        
            if self._saving_falg==True:
                self.save_images_processed(subfolder_paths[0], image_process_learning_batched, trajs_learning_batched, proces=True)
                self.save_images_processed(subfolder_paths[1], pns, trajs_learning_batched)
                self.save_images_processed(subfolder_paths[2], kcs, trajs_learning_batched,mask_bin=True)
                self.save_images_processed(subfolder_paths[3], sws, trajs_learning_batched,mappable=True)

            print(f"Training MB_{mb_nb} - Batch {batch_idx + 1}/{num_batches}")

        compressed_memory_final = np.array(compressed_memory)
        
        el = time.time()
        print(f"\nMB_{mb_nb} trained in {el-sl:.2f}sec with {imgs_nb} images")
        print("\n** Training MBs finished **")

        mapping_learning = np.concatenate((trajs_learning_list, unfamiliarities_learning_list, kc_saturation), axis=1)

        return mapping_learning, compressed_memory_final
    
    def test(self,src_image_test,traj_image_trained, batch_size=20):
        filenames = os.listdir(src_image_test)
        num_images = len(filenames)
        num_batches = (num_images + batch_size - 1) // batch_size

        unfamiliarities_exploit_list = np.zeros((num_images,self._mbs_nb), dtype=float)
        trajs_exploit_list = np.zeros((num_images,4), dtype=float)
        XTE = np.zeros((num_images,1), dtype=float)
        HE = np.zeros((num_images,1), dtype=float)

        subfolders = ["kcs0","kcs1","processed", "pns"]
        subfolder_paths = [os.path.join(self._outputfolder, f"images/test/{subfolder}/") for subfolder in subfolders]   

        sl = time.time()
        for batch_idx in range(num_batches):
        # Process images in batches
        # for start_idx in range(0, len(images_route_all), batch_size):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_images)

            images_exploit_batched, trajs_exploit_batched = self.read_raw_img_folder(src_image_test,False,start_idx,end_idx)

            pns_exploit_batched,image_process_exploit_batched = self.create_pnbank(images_exploit_batched)

            XTE[start_idx:end_idx,0], HE[start_idx:end_idx,0] = self.calcul_cross_track_errors(traj_image_trained[:, 1:4],trajs_exploit_batched[:, 1:4])
            
            trajs_exploit_list[start_idx:end_idx,:] = trajs_exploit_batched

            for it in range(self._mbs_nb):
                unfamiliarities_exploit_list[start_idx:end_idx,it], pns, kcs = self.compare(pns_exploit_batched, self._mbs[it])            
            
            if self._saving_falg:
                self.save_images_processed(subfolder_paths[it], kcs, trajs_exploit_batched,mask_bin=True)
                self.save_images_processed(subfolder_paths[2], image_process_exploit_batched, trajs_exploit_batched,proces= True)
                self.save_images_processed(subfolder_paths[3], pns, trajs_exploit_batched)
            
            print(f"Exploitation MBs - Batch {batch_idx + 1}/{num_batches}")
        el = time.time()

        print("\n** Exploitation of MBs finished **")
        print(f"-> {num_images} images tested in {el-sl:.2f}sec\n")

        mapping = np.concatenate((trajs_exploit_list,unfamiliarities_exploit_list, XTE, HE), axis=1)

        return mapping

    def compare(self, pn_viewbank, mb):
        unfamiliarities = []
        pns_list = []
        kcs_list = []
        sl = time.time()
        for idx, pn in enumerate(pn_viewbank):
            unfamiliarities.append(mb.get_unfamiliarity(pn))
            pns = mb.pn_activity
            pns_list.append(pns.copy())
            kcs_list.append(mb.kc_spikes)
            del pn  # Clear the reference to free memory
        el = time.time()
        # print(f"Comparison of {idx} views in {el-sl:.2f}s")
        return unfamiliarities,np.array(pns_list),np.array(kcs_list)

    def calcul_cross_track_errors(self,traj_train,exploit_traj):
        # print("Calculing Cross Track Errors")
        XTEs = []
        HEs = []
        for i in range(len(exploit_traj)):
            distances = np.sqrt(
                (traj_train[:, 0] - exploit_traj[i, 0]) ** 2
                + (traj_train[:, 1] - exploit_traj[i, 1]) ** 2
            )

            min_index = np.argmin(distances)
            nearest_point_learn = traj_train[min_index, :2]

            # Calculate and display the distance and difference in theta
            XTE = distances[min_index]
            HE = traj_train[min_index, 2] + exploit_traj[i, 2] #            HE = traj_train[min_index, 2] - exploit_traj[i, 2]

            # print(f' learn = {learned_traj[min_index, 2]} and expl = {exploit_traj[i,2]}')

            # Calculate the vector from the point on the path to the current point
            vector_to_current_point = [
                exploit_traj[i, 0] - nearest_point_learn[0],
                exploit_traj[i, 1] - nearest_point_learn[1],
            ]

            # Calculate the direction vector of the path
            path_direction = [
                np.cos(traj_train[min_index, 2]),
                np.sin(traj_train[min_index, 2]),
            ]

            # Calculate the cross product to determine the side of the path
            cross_product = (
                path_direction[0] * vector_to_current_point[1]
                - path_direction[1] * vector_to_current_point[0]
            )

            # Determine whether the point is on the right or left
            if cross_product > 0:
                XTE = abs(XTE)
            elif cross_product < 0:
                XTE = -abs(XTE)
                
            HE = self.sawtooth(HE)
            XTEs.append(XTE)
            HEs.append(HE)


        return np.array(XTEs), np.array(HEs)

    def save_images_processed(self, subfolder_path, img, trajs, mask_bin = False, mappable = False, proces = False):
            # Create subfolders if they don't exist
            os.makedirs(subfolder_path, exist_ok=True)

            norm = Normalize(vmin=0, vmax=1)
            scalar_mappable = cm.ScalarMappable(norm=norm, cmap='Reds')
            sh = img.shape
            size = int(np.ceil(np.sqrt(sh[1])))

            for idx, img in enumerate(img):

                # Generate image name
                image_name = f"{trajs[idx,0]}_{trajs[idx,1]}_{trajs[idx,2]}_{trajs[idx,3]}.png"

                # Generate image paths
                image_paths = os.path.join(subfolder_path, image_name)
                # Generate binary mask
                if mask_bin:
                    img = np.where(img.reshape(size, -1) == 1, 0, 255).astype(np.uint8)

                elif mappable:
                    colored_image = scalar_mappable.to_rgba(img.reshape(size, -1), bytes=True)
                    img = cv2.cvtColor(colored_image, cv2.COLOR_RGBA2BGR)

                elif proces:
                    # if (int(self._params_prune[0])):
                    #     # masked_imaged = img.copy()
                    #     img[self._prun_ims_mask == 0] = 0  # Assuming first_image is in BGR format
                    pass
                # Save images
                cv2.imwrite(image_paths, img)
