import math

# 超参数搜索方法，可以选择：grid random bayes
sweep_config = {
    'method': 'grid'
    }

# 参数范围
parameters_dict = {
    'optimizer': {
        'values': ['adam']
    },
    'learning_rate': {
        # # a flat distribution between 0 and 0.1
        # 'distribution': 'uniform',
        # 'min': 0,
        # 'max': 0.1
        'values': [0.001]
    },
    'batch_size': {
        # # integers between 32 and 256
        # # with evenly-distributed logarithms
        # 'distribution': 'q_log_uniform',
        # 'q': 1,
        # 'min': math.log(32),
        # 'max': math.log(256),
        'values': [256]
    },
    'feature_num': {
        'values': [2,3,4,5,6,7,8]
      }
    }

sweep_config['parameters'] = parameters_dict