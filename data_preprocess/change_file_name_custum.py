import os
from configuration import Config
import shutil


dataset_path = Config.base_path + 'custum_dataset' # 'custum_dataset''augmentation_dataset'

for user in ["user_05"]: #Config.user_list:
    for type in Config.target_list:
        file_path = dataset_path +'/' + user+ '/' + type
        file_names = os.listdir(file_path)
        i = 1
        for name in file_names:
            source = file_path + '/' + name
            new = 'ori' + "_" + user + "_" + '{0:04d}'.format(i) + "_"+type + '.wav'
            dst = file_path+'/'+new
            os.rename(source, dst)
            i += 1

#
google_file_path = "C:/Users/yutan/temp_audio/"

for google_type in os.listdir(google_file_path):
    folder_path = google_file_path + google_type
    i = 1
    for name in os.listdir(folder_path):
        if i == 50:
            continue
        start = folder_path + '/' + name
        end = "C:/Users/yutan/Desktop/Wake-up-Word_tensorflow2/augmentation_dataset/user_01/other_google_speech/" + name
        shutil.move(start, end)
        i += 1

        # path = folder_path +'/' + name
        #
        # new_name = "other_google_speech"+'_'+"user_01" + '_'  + '{0:04d}'.format(i) + google_type + '.wav'
        # new_name_path = folder_path + '/'+ new_name
        # os.rename(path, new_name_path)
        # i += 1
