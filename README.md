# 1. Generate Backdoor dataset
In poison_data_generator directory
poison_percentage: the poison ratio in single class, the suffix of the output path must be .npz

RUN cd poison_data_generator

RUN python badnet.py --poison_percentage=0.05 --trainData_output_path="../data/badnet_5%.npz" --cleanData_output_path="../data/clean_data.npz" --poisonData_output_path="../data/poison_data.npz"

# 2. If your FFCV ENV doesn't works well
## the sampling_datasize must lower than 5000
## still debug
RUN python observe_MI.py --observe_class=1 --output_dir="result/today" --training_epochs=100 --sampling_datasize=4000

## If your FFCV ENV works good.

RUN python ffcv_writer.py --output_path=train_data.beton --dataset=train_dataset --train_data_path=xxxxx --test_data_path=xxxx  --train_cleandata_path=xxxxx --train_poisondata_path=xxxxxx

RUN python ffcv_writer.py --output_path=test_data.beton --dataset=test_dataset --train_data_path=xxxxx --test_data_path=xxxxx --train_cleandata_path=xxxxx --train_poisondata_path=xxxxx

RUN python ffcv_writer.py --sampling_datasize=4000 --observe_class=0 --output_path=observe_data.beton --train_data_path=xxxxxx --test_data_path=xxxxx --dataset=sample_dataset --train_cleandata_path=xxxxx --train_poisondata_path=xxxxx

# 3. RUN FFCV_observeMI.py
RUN python ffcv_observeMI.py --train_data_path="xxx.beton" --test_data_path="xxx.beton" --sample_data_path="xxx.beton"

# 4. RUN plot.py
RUN python plot.py --input_output_MI_path="xxx.npy" --output_modelOutput_MI_path="xxx.npy"
