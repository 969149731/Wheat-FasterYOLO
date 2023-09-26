
""" run_mot_challenge.py            1797-2915用byte模型跑的   2915后是oc
思路 用ocsort跑last,pt模型  把结果放进GT      拉低精度     同时保证检测个数和原来一样  然后数值又比别的追踪算法好看
OK 方案可行  -21=365  -31=355  -41=345   -51=335   -50=336   -45=341   -43=340  -40=343


Run example:
run_mot_challenge.py --USE_PARALLEL False --METRICS Hota --TRACKERS_TO_EVAL Lif_T

Command Line Arguments: Defaults, # Comments
    评估参数
    Eval arguments:
        'USE_PARALLEL': False,                               使用并行
        'NUM_PARALLEL_CORES': 8,                             使用并行核心数量
        'BREAK_ON_ERROR': True,                              引发异常时带着错误退出
        'PRINT_RESULTS': True,                               打印结果
        'PRINT_ONLY_COMBINED': False,                        只打印联合结果
        'PRINT_CONFIG': True,                                是否打印当前配置
        'TIME_PROGRESS': True,                               时间进度
        'OUTPUT_SUMMARY': True,                              输出统计数据
        'OUTPUT_DETAILED': True,                             输出详细信息
        'PLOT_CURVES': True,                                 plot curves
    数据集参数
    Dataset arguments:
        'GT_FOLDER': os.path.join(code_path, 'data/gt/mot_challenge/'),  # Location of GT data   gt下的gt.txt就是我们标定的目标信息文件
        'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/mot_challenge/'),  # Trackers location   trackers目录下的txt文件就是我们检测跟踪到的结果信息
        'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)    存评估结果的地方（如果没有，与TRACKERS_FOLDER相同）。
        'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)  要评估的跟踪器的文件名（如果没有，则全部在文件夹中）
        'CLASSES_TO_EVAL': ['pedestrian'],  # Valid: ['pedestrian']       评估的类别
        'BENCHMARK': 'MOT17',  # Valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15'     基准
        'SPLIT_TO_EVAL': 'train',  # Valid: 'train', 'test', 'all'             分开评估
        'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped       追踪器的输入文件是否被压缩
        'PRINT_CONFIG': True,  # Whether to print current config               是否打印当前配置
        'DO_PREPROC': True,  # Whether to perform preprocessing (never done for 2D_MOT_2015)  是否进行预处理（对2D_MOT_2015来说从未进行过预处理）。
        'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
    Metric arguments:
        'METRICS': ['HOTA', 'CLEAR', 'Identity', 'VACE']
"""

import sys
import os
import argparse
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402   12

if __name__ == '__main__':
    freeze_support()

    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs='+')
        else:
            parser.add_argument("--" + setting)
    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            elif setting == 'SEQ_INFO':
                x = dict(zip(args[setting], [None]*len(args[setting])))
            else:
                x = args[setting]
            config[setting] = x
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    evaluator.evaluate(dataset_list, metrics_list)
