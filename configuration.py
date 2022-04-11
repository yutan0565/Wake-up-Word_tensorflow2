from os import listdir
from os.path import isdir, join

class Config:

    # base_path = "/content/drive/MyDrive/"
    #base_path = "C:/Users/yutan/Desktop/Wake-up-Word_tensorflow2/"
    base_path = "C:/Users/yutankim/Desktop/Wake-up-Word_tensorflow2/"

    for_aug_dataset = 'custum_dataset'  #Augmentation을 적용할 폴더 이름
    dataset_type = "augmentation_dataset"      # 학습에 사용할 데이터 셋

    aug_dataset_path = base_path + for_aug_dataset   # augmentation을 진행 할 곳
    dataset_path =  base_path + dataset_type  # 학습에 사용할 데이터셋 경로

    print(aug_dataset_path)


    # 데이터 나누는 비율
    val_ratio = 0.1
    test_ratio = 0.2
    
    # Augmentation 종류 별, 본래 클래스 비율
    aug_rate = 0.9

    target_list = ['hey_tantan', 'hi_byeonghyeon','hi_jeonglyul', 'hi_sungwoo', 'hi_yutan', 'other', 'other_google_speech'] #, 'shift_sound'
    user_list = ['user_01', 'user_02', 'user_03', 'user_04', 'user_05']

    # [김유탄, 성우, 병현]
    # 기동어 class 이름 - 폴더이름
    target_wake_word = ['hi_yutan']
    target_user = ['user_01']

    target_wake_word_index = [4]
    target_user_index = [0]

    # 기동어 정답 나누는 기준
    thres_hold_detect = 0.98
    thres_hold_recog = 0.80

    # thres_hold_low_power = 0.002
    thres_hold_low_power = 0.001


    # 음향 파일 불러올떄 1초당 sample 개수
    sample_rate = 32000
    stride_rate = 8

    low_power_sample_rate = 4000

    wav_time = 1
    sample_cut = int(sample_rate*wav_time)
    click = int(sample_rate*0.2)

    n_fft = 1024
    hop_length = 512
    win_length = 1024

    n_mels = 256
    len_mfcc = 63


    num_mfcc=13
    
    # 학습 관련
    epoch_original = 300
    batch_size_original = 64
    early_stop_aptience = 30
    lr_factor = 0.7
    lr_patience = 10
    start_lr = 0.0001

    #Pruning 관련
    epoch_prun = 300
    batch_size_prun = 32

    # baes model Checkpoint 경로
    best_model_path = base_path+ "best_model/wake_up_word_model"
    best_model_path_recog = base_path+ "best_model/speaker_recognition_model"
    best_model_path_recog_02 = base_path + "best_model/speaker_recognition_model_02"


    best_model_path_detect_pruning_06 = base_path+ "best_model/detection_pruning_06_model"
    best_model_path_detect_pruning_08 = base_path+ "best_model/detection_pruning_08_model"
    best_model_path_detect_pruning_09 = base_path+ "best_model/detection_pruning_09_model"
    best_model_path_detect_pruning_95 = base_path+ "best_model/detection_pruning_95_model"
    best_model_path_detect_pruning_98 = base_path+ "best_model/detection_pruning_98_model"
    best_model_path_detect_pruning_99 = base_path+ "best_model/detection_pruning_99_model"


    best_model_path_recog_pruning_06_02 = base_path + "best_model/recognition_pruning_model_06_02"
    best_model_path_recog_pruning_08_02 = base_path + "best_model/recognition_pruning_model_08_02"
    best_model_path_recog_pruning_09_02 = base_path + "best_model/recognition_pruning_model_09_02"
    best_model_path_recog_pruning_95_02 = base_path + "best_model/recognition_pruning_model_95_02"
    best_model_path_recog_pruning_98_02 = base_path + "best_model/recognition_pruning_model_98_02"
    best_model_path_recog_pruning_99_02 = base_path + "best_model/recognition_pruning_model_99_02"



    #기본 tfltie 파일 저장
    tflite_model_path = 'tflite_converter/tflite_model/'

    # Detect
    tflite_file_path = base_path + tflite_model_path + 'ori_detection.tflite'
    quant_tflite_file_path = base_path + tflite_model_path+ 'quant_detection.tflite'


    prun_06_tflite_file_path = base_path + tflite_model_path + 'prun_06_detection.tflite'
    prun_08_tflite_file_path = base_path + tflite_model_path + 'prun_08_detection.tflite'
    prun_09_tflite_file_path = base_path + tflite_model_path + 'prun_09_detection.tflite'
    prun_95_tflite_file_path = base_path + tflite_model_path + 'prun_95_detection.tflite'
    prun_98_tflite_file_path = base_path + tflite_model_path + 'prun_98_detection.tflite'
    prun_99_tflite_file_path = base_path + tflite_model_path + 'prun_99_detection.tflite'


    # Recog
    # tflite_file_path_recog = base_path + tflite_model_path+ 'ori_recognition.tflite'
    tflite_file_path_recog_02 = base_path + tflite_model_path + 'ori_recognition_02.tflite'
    quant_tflite_file_path_recog = base_path + tflite_model_path+ 'quant_recognition.tflite'


    prun_06_tflite_file_path_recog = base_path + tflite_model_path + 'prun_06_recognition.tflite'
    prun_08_tflite_file_path_recog = base_path + tflite_model_path + 'prun_08_recognition.tflite'
    prun_09_tflite_file_path_recog = base_path + tflite_model_path + 'prun_09_recognition.tflite'
    prun_95_tflite_file_path_recog = base_path + tflite_model_path + 'prun_95_recognition.tflite'
    prun_98_tflite_file_path_recog = base_path + tflite_model_path + 'prun_98_recognition.tflite'
    prun_99_tflite_file_path_recog = base_path + tflite_model_path + 'prun_99_recognition.tflite'








    


