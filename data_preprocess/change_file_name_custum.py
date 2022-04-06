import os
from configuration import Config


base_path = "../"
dataset_path = base_path + 'custum_dataset' # 'custum_dataset''augmentation_dataset'

for user in Config.user_list:
    for type in Config.target_list:
        file_path = './custum_dataset/' + user+ '/' + type
        file_names = os.listdir(file_path)
        i = 1
        for name in file_names:
            source = file_path + '/' + name
            new = 'ori' + "_" + user + "_" + '{0:04d}'.format(i) + "_"+type + '.wav'
            dst = file_path+'/'+new
            os.rename(source, dst)
            i += 1