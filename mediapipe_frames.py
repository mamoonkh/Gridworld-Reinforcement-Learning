import mediapipe as mp
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import logging
from aicurelib.continuous_learning import aicure_video_download


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

logging.basicConfig(level=logging.INFO)


def parse_args():
    
    """ This function creates an argument parser and parses the input arguments """
    
    parse = argparse.ArgumentParser(description = 'Runs mediapipe on TD videos')
    
    parse.add_argument("--video_urls", 
                       required=True,
                       nargs='+',
                       help = "List of Video urls. This follows the format: "
                      r"prod-cortex-secure.aicure.com/videos/dbm_external_data/neurocrine_td/Study_{1304/1402}/{id}.mp4")
    
    parse.add_argument("--output_folder",
                       type = str, 
                       required=True,
                       help = "Folder where to store the pose estimation landmarks and videos.")
    
    parse.add_argument("--draw_pose", 
                       default = False,
                       help = "Optional to save the pose estimation videos.")
    
    
    args = parse.parse_args()
    
    return args
    
    

def download_video(video_url, output_folder):
    
    data={'aicure_master_video_url': video_url}

    try:
        path = aicure_video_download.download_dbm_video(data, output_folder)
    except:
        raise Exception("Video Cannot be downloaded")

    return path
    
    

def get_video_id(video_url):
    
    video_split = video_url.split('/')
    video_id = video_split[-2] + '/' + video_split[-1]
    
    return video_id



def run_mediapipe_pose_estimation(downloaded_video_path, output_folder, video_id, draw_pose=False):
   
    landmarks_list = []
    
    cap = cv2.VideoCapture(downloaded_video_path)
    
    frame_width =  int(cap.get(3))
    frame_height =  int(cap.get(4))
    
    frame_count = 0
    
    
    if draw_pose:
         
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter()
        
        pe_path = output_folder + '/pose_estimation_videos/' + video_id
        
        try:
            os.makedirs(os.path.split(pe_path)[0])
        except:
            pass
        
        success = out.open(pe_path, fourcc, 20, (frame_width, frame_height), True)
        
        
    with mp_pose.Pose() as pose:

        while cap.isOpened():

            ret, frame = cap.read()

            if ret == True:

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                try:
                    results = pose.process(image)
                    landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark]
                    landmarks_list.append(landmarks)
                except:
                    continue

                if draw_pose:

                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    try:
                        mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                        out.write(image)
                    except:
                        continue
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
#             frame_count +=1
#             if frame_count==100:
#                 break
            
                    
    return np.array(landmarks_list), frame_width, frame_height


def main():
    
    args = parse_args()
    
    logging.info('Running Script')
    
    df = pd.DataFrame(columns = ["aicure_master_video_url", "frame_width", "frame_height"])
    index = 0
    
    for video_url in args.video_urls:
        
        
        downloaded_video_path = download_video(video_url, args.output_folder)
        logging.info("Original TD video downloaded to: {}".format(downloaded_video_path))

        video_id = get_video_id(video_url)
        
        landmarks_array, width, height = run_mediapipe_pose_estimation(downloaded_video_path, args.output_folder, video_id, draw_pose=args.draw_pose)

        output_file_path = os.path.join(args.output_folder, 'mediapipe_frames/' +  video_id + ".npy")

        try:
            os.makedirs(os.path.split(output_file_path)[0])
        except:
            pass

        np.save(output_file_path, landmarks_array)
        logging.info('Mediapipe Pose Estimation Frames File Saved to: {}'.format(output_file_path)) 
        
        
        df.loc[index, "aicure_master_video_url"] = video_url
        df.loc[index, "frame_width"] = width
        df.loc[index, "frame_height"] = height
        index+=1
        
    df.to_csv(args.output_folder + "/results.csv")
    logging.info('Saving the results dataframe to: {}'.format(output_file_path)) 
        
    
    
if __name__ == '__main__':
    
    main()
