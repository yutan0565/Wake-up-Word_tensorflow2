from os import listdir
from os.path import isdir, join

class Config:

    # 코랩 또는 그냥 컴퓨터
    # base_path = "/content/drive/MyDrive/"
    base_path = "C:/Users/yutan/Desktop/Wake-up-Word_tensorflow2/"
    
    #Augmentation을 적용할 폴더 이름
    for_aug_dataset = 'custum_dataset'
    
    # 학습에 사용할 데이터 셋
    dataset_type = "augmentation_dataset"
    

    # augmentation을 진행 할 곳
    aug_dataset_path = base_path + for_aug_dataset

    # 학습에 사용할 데이터셋 경로
    dataset_path =  base_path + dataset_type

    print(aug_dataset_path)


    # 데이터 나누는 비율
    val_ratio = 0.1
    test_ratio = 0.2
    
    # Augmentation 종류 별, 본래 클래스 비율
    aug_rate = 0.9
    
    # 내가 신경 안쓸 class
    non_target_list = ['_background_noise_', '.ipynb_checkpoints' ]

    target_list = ['hey_tantan', 'hi_byeonghyeon','hi_jeonglyul', 'hi_sungwoo', 'hi_yutan', 'other'] #, 'shift_sound'

    user_list = ['user_01', 'user_02', 'user_03']

    # 기동어 class 이름 - 폴더이름
    target_wake_word = ['hi_yutan', 'hi_sungwoo']
    target_user = ['user_01', 'user_02']

    target_wake_word_index = [3, 4]
    target_user_index = [0, 1]

    # 음향 파일 불러올떄 1초당 sample 개수
    sample_rate = 32000
    low_power_sample_rate = 4000

    # 기동어 정답 나누는 기준
    thres_hold_detect = 0.98
    thres_hold_recog = 0.8
    thres_hold_low_power = 0.0001


    # 내가 잘라내고 싶은 총 시간
    wav_time = 1 #초단위
    #sample rate 고려한 자르는 시작점
    sample_cut = int(sample_rate*wav_time) 
    # 클릭 시간
    click = int(sample_rate*0.1)
    # frame 몇개로 할지 선정
    num_mfcc=13
    #
    n_fft = 1024
    hop_length = 512
    win_length = 1024

    n_mels = 256
    len_mfcc = 63

    stride_rate = 4
    
    
    # 학습 관련
    epoch_original = 300
    batch_size_original = 32
    
    early_stop_aptience = 30
    # baes model Checkpoint 경로
    best_model_path = base_path+ "best_model/wake_up_word_model"
    best_model_path_recog = base_path+ "best_model/speaker_recognition_model"
    best_model_path_recog_02 = base_path + "best_model/speaker_recognition_model_02"

    best_model_path_detect_pruning = base_path+ "best_model/detection_pruning_model"
    best_model_path_recog_pruning = base_path+ "best_model/recognition_pruning_model"
    lr_factor = 0.7
    lr_patience = 10
    start_lr = 0.0001

    tflite_model_path = 'tflite_converter/tflite_model/'

    #기본 tfltie 파일 저장 
    tflite_file_path = base_path + tflite_model_path + 'ori_detection.tflite'
    #Quantization tflite 파일 저장
    quant_tflite_file_path = base_path + tflite_model_path+ 'quant_detection.tflite'
    #Pruning tflite 파일 저장
    prun_tflite_file_path = base_path + tflite_model_path+ 'prun_detection.tflite'

    tflite_file_path_recog = base_path + tflite_model_path+ 'ori_recognition.tflite'

    tflite_file_path_recog_02 = base_path + tflite_model_path + 'ori_recognition_02.tflite'
    #Quantization tflite 파일 저장
    quant_tflite_file_path_recog = base_path + tflite_model_path+ 'quant_recognition.tflite'
    #Pruning tflite 파일 저장
    prun_tflite_file_path_recog = base_path + tflite_model_path+ 'prun_recognition.tflite'

    
    #Pruning 관련
    batch_size_prun = 32
    epoch_prun = 300
    validation_split_prun = 0.1

