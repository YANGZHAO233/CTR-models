from models.DCN import DCN
from models.DCN_V2_Parallel import DCN_V2_Parallel
from models.DCN_V2_Stack import DCN_V2_Stack
from models.DCN_Attention_Parallel_V1 import DCN_Attention_Parallel_V1
from models.DCN_Attention_Parallel_V2 import DCN_Attention_Parallel_V2
from models.DCN_Attention_Stack_V1 import DCN_Attention_Stack_V1
from models.DCN_Attention_Stack_V2 import DCN_Attention_Stack_V2
from models.xDeepFM import xDeepFM
from models.xDeepFM_Attention import xDeepFM_Attention
from models.DCN_tf import DCN_tf


def build_models(feature_columns, params, model_name='DCN'):
    if model_name == 'DCN':
        model = DCN(
            feature_columns,
            hidden_units=params['hidden_units'],
            output_dim=params['output_dim'],
            activation='relu',
            layer_num=params['layer_num']
        )
        return model

    elif model_name == 'DCN_V2_Parallel':
        model = DCN_V2_Parallel(
            feature_columns,
            hidden_units=params['hidden_units'],
            output_dim=params['output_dim'],
            activation='relu',
            layer_num=params['layer_num']
        )
        return model

    elif model_name == 'DCN_V2_Stack':
        model = DCN_V2_Stack(
            feature_columns,
            hidden_units=params['hidden_units'],
            output_dim=params['output_dim'],
            activation='relu',
            layer_num=params['layer_num']
        )
        return model

    elif model_name == 'DCN_Attention_Parallel_V1':
        model = DCN_Attention_Parallel_V1(
            feature_columns,
            hidden_units=params['hidden_units'],
            output_dim=params['output_dim'],
            activation='relu',
            layer_num=params['layer_num']
        )
        return model

    elif model_name == 'DCN_Attention_Parallel_V2':
        model = DCN_Attention_Parallel_V2(
            feature_columns,
            hidden_units=params['hidden_units'],
            output_dim=params['output_dim'],
            activation='relu',
            layer_num=params['layer_num']
        )
        return model

    elif model_name == 'DCN_Attention_Stack_V1':
        model = DCN_Attention_Stack_V1(
            feature_columns,
            hidden_units=params['hidden_units'],
            output_dim=params['output_dim'],
            activation='relu',
            layer_num=params['layer_num']
        )
        return model

    elif model_name == 'DCN_Attention_Stack_V2':
        model = DCN_Attention_Stack_V2(
            feature_columns,
            hidden_units=params['hidden_units'],
            output_dim=params['output_dim'],
            activation='relu',
            layer_num=params['layer_num']
        )
        return model

    elif model_name == 'xDeepFM':
        model = xDeepFM(
            feature_columns,
            hidden_units=params['hidden_units'],
            output_dim=params['output_dim'],
            activation='relu',
            cin_size=params['cin_size']
        )
        return model

    elif model_name == 'xDeepFM_Attention':
        model = xDeepFM_Attention(
            feature_columns,
            hidden_units=params['hidden_units'],
            output_dim=params['output_dim'],
            activation='relu',
            cin_size=params['cin_size']
        )
        return model

    elif model_name == 'DCN_tf':
        model = DCN_tf(
            feature_columns,
            hidden_units=params['hidden_units'],
            output_dim=params['output_dim'],
            activation='relu',
            layer_num=params['layer_num']
        )
        return model

    else:
        print("Error model input!")
        return None