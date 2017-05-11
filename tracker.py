#importing required packages
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# Define a class to receive the characteristics of each line detection
class tracker():
    def __init__(self, Mywindow_width, Mywindow_height, Mymargin, My_ym = 1, My_xm = 1, Mysmooth_factor = 15):
        
        #list that stores all the past (left, right) center set values used for smoothing the o/p
        self.recent_centers = []
        
        #the window pixel width of the center values, used to count pixels inside center windows to determine curve values
        self.window_width = Mywindow_width
        
        #the window pixel height of the center values, used to count pixels inside center windows to determine curve values
        self.window_height = Mywindow_height
        
        # The pixel distance in both directions to slide (left_window + right_window) template for searching (padding)
        self.margin = Mymargin
        
        self.ym_per_pix = My_ym # meters per pixel in vertical axis

        self.xm_per_pix = My_xm # meters per pixel in horizontal axis
        
        # How many previous best curve sets to average over to get smooth results.
        self.smooth_factor = Mysmooth_factor
        
    # main tracking function for finding and storing lane segment positions    
    def find_window_centroids(self, wraped_img):
        wraped = wraped_img
        #print("wraped image size:", wraped.shape)
        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin
        #print("window_width:",window_width)
        window_centorids = [] 
        window = np.ones(window_height)
        #print("window", window)
        if (len(self.recent_centers) == 0):
            #print("INITIAL")
            l_sum = np.sum(wraped[int(3*wraped.shape[0]/4):,:int(wraped.shape[1]/2)],axis=0)
            l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
            r_sum = np.sum(wraped[int(3*wraped.shape[0]/4):,int(wraped.shape[1]/2):],axis=0)
            r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(wraped.shape[1]/2)
        else :
            #print("Second Round")
            r_center = self.recent_centers[-1][0][1]
            l_center = self.recent_centers[-1][0][0]

            # sliding around previous start point
            l_sum = np.sum(wraped[int(3 * wraped.shape[0] / 4):, l_center - self.margin:l_center + self.margin], axis=0)
            if len(l_sum) > 0:
                l_sum = l_sum * np.concatenate((np.linspace(.5, 1, self.margin), np.linspace(1, .5, self.margin)),axis=0)
                if (np.max(l_sum) > 100):
                    l_center = int(np.argmax(l_sum)) + (l_center - self.margin)
            else :
                #print("Second Round Back to Initial")
                l_sum = 0
                l_sum = np.sum(wraped[int(3*wraped.shape[0]/4):,:int(wraped.shape[1]/2)],axis=0)
                #l_center = np.argmax(np.convolve(window,l_sum))-window_width / 2
                l_center = np.argmax(l_sum)


            # for the right start
            r_sum = np.sum(wraped[int(3 * wraped.shape[0] / 4):, r_center - self.margin:r_center + self.margin], axis=0)
            if len(r_sum) > 0:
                r_sum = r_sum * np.concatenate((np.linspace(.5, 1, self.margin), np.linspace(1, .5, self.margin)),axis=0)
                if ((np.max(r_sum) > 100) & (r_sum.shape == 50)):
                    r_center = int(np.argmax(r_sum)) + (r_center - self.margin)
            else :
                r_sum = 0
                r_sum = np.sum(wraped[int(3*wraped.shape[0]/4):,int(wraped.shape[1]/2):], axis=0)
                #r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(wraped.shape[1]/2)
                r_center = np.argmax(r_sum) + int(wraped.shape[1] / 2)

        # Add what we found for the first layer 
        window_centorids.append((l_center,r_center))
        
        # Similarly go through each layer looking for max pixel location
        for level in range(1,(int)(wraped.shape[0]/window_height)):
            
            # Convolve the window into veritical slice of the image
            #image_layer1 = wraped[int(wraped.shape[0]-(level+1)*window_height):int(wraped.shape[0]-level*window_height),:]
            #print("image_layer1", image_layer1)

            image_layer = np.sum(wraped[int(wraped.shape[0]-(level+1)*window_height):int(wraped.shape[0]-level*window_height),:], axis = 0)
            #print("for loop image_layer", image_layer)
            conv_signal = np.convolve(window, image_layer)
            #print("conv_signal", conv_signal)
            #Find best left centroid using the past left center as reference
            #Use window_width/2 as offset because convolvution signal reference is at the right side of the window, not at center
            offset = window_width/2

            #print("l_min_index", l_center+offset-margin)
            #print("l_max_index", l_center+offset+margin)

            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center + offset + margin, wraped.shape[1]))
            #print("conv_signal[l_min_index:l_max_index]",conv_signal[l_min_index:l_max_index])
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
            
            #Find the best Right centroid
            #r_min_index = int(max(r_center + offset - margin, 0))
            #r_max_index = int(min(r_center + offset + margin, wraped.shape[1]))
            r_min_index = int(max(r_center + offset - margin, 0))
            r_max_index = int(min(r_center + offset + margin, wraped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index - offset
            
            #Add to this layer Window centorids
            window_centorids.append((l_center, r_center))
            
        self.recent_centers.append(window_centorids)
        #print("self.recent_centers", self.recent_centers)


        #return self.recent_centers
        return np.average(self.recent_centers[-self.smooth_factor:], axis=0)