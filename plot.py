import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import scipy.stats as stats


def plot_info_plane():
    # inputs_outputs_arr = np.load(args.input_output_MI_path
    #                               , allow_pickle=True)
    # Y_outputs_arr = np.load(args.output_modelOutput_MI_path
    #                          , allow_pickle=True)

    # print(inputs_outputs_arr_clean1.shape, Y_outputs_arr_clean1.shape)  # (11, ) (11, 100)
    inputs_outputs_arr = [1.966810941696167, 1.9570984840393066, 2.0179378986358643, 2.0758488178253174, 2.064568519592285,
                          1.9758517742156982, 1.9844354391098022, 1.8858222961425781, 1.8947159051895142, 1.8761343955993652,
                          1.8461711406707764, 1.7963922023773193, 1.8015727996826172, 1.7962313556671143, 1.7789636850357056,
                          1.7197344303131104, 1.7478834390640259, 1.7062435150146484, 1.7271174192428589, 1.7208468914031982,
                          1.632126808166504, 1.6786749362945557, 1.711211085319519, 1.61306631565094, 1.6600602865219116,
                          1.637441873550415, 1.6637144088745117, 1.6044862270355225, 1.6363909244537354, 1.614157795906067,
                          1.5809589624404907, 1.586227536201477, 1.6142401695251465, 1.6042401695251465]
    Y_outputs_arr = [0.3306751251220703, 0.445840984582901, 0.4744081497192383, 0.49656954407691956, 0.5558507442474365,
                     0.6326214671134949, 0.6718343496322632, 0.6824291944503784, 0.7401873469352722, 0.7426693439483643,
                     0.7346112132072449, 0.7360308170318604, 0.7525580525398254, 0.7559211850166321, 0.7949938435554504,
                     0.7756568193435669, 0.7834101319313049, 0.7364987730979919, 0.7740195393562317, 0.770911455154419,
                     0.8505931496620178, 0.8077322244644165, 0.7958163619041443, 0.8533579111099243, 0.7871130704879761,
                     0.7927736639976501, 0.8410639762878418, 0.8378931283950806, 0.8284274935722351, 0.7979740500450134,
                     0.8576619029045105, 0.8292196393013, 0.8219774961471558, 0.7896926999092102]
    inputs_outputs_arr = [1.2981112003326416, 1.4175214767456055, 1.448261022567749, 1.3729948997497559, 1.4243298768997192,
                          1.426449179649353, 1.3850303888320923, 1.348482608795166, 1.2840644121170044, 1.2638715505599976,
                          1.2889866828918457, 1.2483021020889282, 1.2470225095748901, 1.2699894905090332, 1.199554681777954,
                          1.2205278873443604, 1.191558599472046, 1.156309723854065, 1.1757705211639404, 1.176129937171936,
                          1.188680648803711, 1.152747631072998, 1.171494483947754, 1.1609950065612793, 1.1168241500854492,
                          1.1212177276611328, 1.1289616823196411, 1.1062383651733398, 1.1676136255264282, 1.121880054473877,
                          1.1048684120178223, 1.101332426071167, 1.0822157859802246, 1.0822157859802246]
    info_plane = np.empty([34, 2])
    for idx in range(34):
        info_plane[idx, 0] = np.mean(inputs_outputs_arr[idx])
        info_plane[idx, 1] = np.mean(Y_outputs_arr[idx])
    # x轴: inputs, Y轴: outputs
    df = pd.DataFrame(columns=['Epoch', 'I(X;T)', 'I(T;Y)'])
    df['Epoch'] = np.arange(0, 100, 3)
    df['I(X;T)'] = info_plane[:, 0]
    df['I(T;Y)'] = info_plane[:, 1]
    fig, ax = plt.subplots()
    sca = ax.scatter(x=df['I(X;T)'], y=df['I(T;Y)'], c=df['Epoch'], cmap='summer')
    ax.set_xlabel('I(X;T)')
    ax.set_ylabel('I(T;Y)')
    fig.colorbar(sca, label="Epoch", orientation="vertical")
    fig.savefig("info_plane.png", dpi=300)
    # 从文件中加载图像
    image = plt.imread("info_plane.png")

    # 显示图像
    plt.imshow(image)
    plt.axis('off')  # 可选：去除坐标轴
    plt.show()

def test():
    arr = np.load(r'figs-inputs-vs-outputs/infoNCE.npy')
    print(arr.shape)
    pass

# inputs_outputs_arr = np.load(args.input_output_MI_path
#                                   , allow_pickle=True)
#     Y_outputs_arr = np.load(args.output_modelOutput_MI_path
#                              , allow_pickle=True)
#     print(inputs_outputs_arr.shape)
#     print(Y_outputs_arr.shape)
#     # print(inputs_outputs_arr_clean1.shape, Y_outputs_arr_clean1.shape)  # (11, ) (11, 100)
#     info_plane = np.empty([33, 2])
#     for idx in range(33):
#         info_plane[idx, 0] = np.mean(inputs_outputs_arr[idx][-5:])
#         info_plane[idx, 1] = np.mean(Y_outputs_arr[idx][-5:])
#     # x轴: inputs, Y轴: outputs
#     df = pd.DataFrame(columns=['Epoch', 'I(X;T)', 'I(T;Y)'])
#     df['Epoch'] = np.arange(0, 99, 3)
#     df['I(X;T)'] = np.arange(-33, 66, 3)
#     df['I(T;Y)'] = np.power(df['I(X;T)'], 3)
#     # df['I(X;T)'] = info_plane[:, 0]
#     # 根据 y = x的3次方 生成y
#     # df['I(T;Y)'] = info_plane[:, 1]
#
#     record = pd.DataFrame(columns=['label','initial_x','initial_y','turning_x','turning_y','conv_x','conv_y','time_to_turning'], dtype='float32')
#
#     # initial point
#     record.loc[0] = [float(args.label), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#     record.loc[0]['initial_x'] = df.iloc[0]['I(X;T)']
#     record.loc[0]['initial_y'] = df.iloc[0]['I(T;Y)']
#
#     # turning point
#     #根据行 遍历df 将I(x;T) 大于 前两行和后两行的
#
#     for i in range(2,len(df) - 2):
#         before_first = df.iloc[i - 2]['I(X;T)']
#         before_second = df.iloc[i - 1]['I(X;T)']
#         cur = df.iloc[i]['I(X;T)']
#         after_first = df.iloc[i + 1]['I(X;T)']
#         after_last = df.iloc[i + 2]['I(X;T)']
#
#         if cur > before_first and cur > before_second and cur > after_first and cur > after_last:
#             print(cur)
#             record.loc[0]['turning x'] = df.iloc[i]['I(X;T)']
#             record.loc[0]['turning y'] = df.iloc[i]['I(T;Y)']
#             record.loc[0]['time to turning'] = df.iloc[i]['Epoch']
#             break
#
#
#     # converge point
#     # 计算倒数4个平均值
#     record.loc[0]['conv_x'] = np.mean(df.iloc[-4:]['I(X;T)'])
#     record.loc[0]['conv_y'] = np.mean(df.iloc[-4:]['I(T;Y)'])
#
#     fig, ax = plt.subplots()
#
#     sca = ax.scatter(x=df['I(X;T)'], y=df['I(T;Y)'], c=df['Epoch'], cmap='summer')
#     sca = ax.scatter(x=record.loc[0]['turning_x'], y=record.loc[0]['turning_y'], c='red', s=50)
#     ax.set_xlabel('I(X;T)')
#     ax.set_ylabel('I(T;Y)')
#     fig.colorbar(sca, label="Epoch", orientation="vertical")
#     fig.savefig("info_plane.png", dpi=300)
#
#     # record 所有的数字 保留两位小数
#     record = record.round(2)
#     file_name = 'record'
#
#     if os.path.exists(file_name + '.csv'):
#         shanghai_tz = pytz.timezone('Asia/Shanghai')
#         file_name = file_name + '_' + datetime.now(shanghai_tz).strftime('%Y%m%d_%H%M%S')
#         record.to_csv(file_name + '.csv')


def manovaAnalyze():
    # 读取数据
    data = pd.read_csv('poison_clean.csv')
    print(data.dtypes)

    exog = data[['label']]
    endog = data[['intial_x', 'initial_y', 'turning_x', 'turning_y', 'conv_x', 'conv_y']].astype(float)

    # 进行MANOVA分析
    maov = MANOVA(endog, exog)
    print(maov.mv_test())


# def annovaAnalyze():
#     # Annova
#     data = pd.read_csv('annova.csv')
#     print(data.dtypes)
#     # annova
#     f_statistic, p_value = stats.f_oneway(data['intial_x'], data['initial_y'], data['turning_x'], data['turning_y'], data['conv_x'], data['conv_y'])
#
#     print('F statistic:', f_statistic)
#     print('P value:', p_value)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input_output_MI_path', type=str, default='results/ob_infoNCE_06_22', help='output_dir')
    # parser.add_argument('--output_modelOutput_MI_path', type=str, default='results/ob_infoNCE_06_22', help='output_dir')
    # args = parser.parse_args()
    plot_info_plane()
    # test()
   # manovaAnalyze()
 #   annovaAnalyze()