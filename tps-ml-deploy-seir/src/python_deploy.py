import boto3
import pandas as pd
from io import StringIO
from scripts.python.models import SEIRBayes
from helper_functions import q25, q975, make_NEIR0, estimate_r0, make_EI_df, make_param_widgets
from constants import SAMPLE_SIZE, MIN_DAYS_r0_ESTIMATE, w_granularity


def main(*args):

    # bucket='todospelasaude-lake-raw-prod' #
    # data_key = 'ExternalSupplier/Covid19BR/Full/Files/cases-brazil-states/cases-brazil-states.csv'
    cases_df_location = 's3://todospelasaude-lake-raw-prod/ExternalSupplier/Covid19BR/Full/Files/cases-brazil-states/cases-brazil-states.csv'
    population_df_location = 's3://todospelasaude-lake-deploys/manual_inputs/ibge_population.csv'

    # local
    # cases_df_location = './cases-brazil-states.csv'
    # population_df_location = './ibge_population.csv'

    cases_df = pd.read_csv(cases_df_location)
    cases_df.loc[cases_df['state'] == 'TOTAL', 'state'] = 'BR'
    cases_df = (
        cases_df.groupby(['date', 'state'])
        [['newCases', 'totalCases']]
            .sum()
            .unstack('state')
            .sort_index()
            .swaplevel(axis=1)
            .fillna(0)
            .astype(int)
    )

    population_df = (
        pd.read_csv(population_df_location)
            .rename(columns={'uf': 'state'})
            .assign(city=lambda df: df.city + '/' + df.state)
            .groupby(w_granularity)
        ['estimated_population']
            .sum()
            .sort_index()
    )

    population_df = population_df.append(pd.Series([population_df.sum()], index=['BR']))

    f = {
        'Exposed': ['mean', 'median', q25, q975],
        'Infected': ['mean', 'median', q25, q975]
    }

    # train_dates = ['2020-05-17', '2020-05-03', '2020-04-28', '2020-04-22', '2020-04-14']
    states = cases_df.columns.get_level_values(0).drop_duplicates().values

    output = pd.DataFrame()

    for state in states:
        # for date in train_dates:
        w_place = state  # STATE / CITY
        w_date = cases_df.index.max() # date  # DATA_TREINO
        t_max = 180

        NEIR0 = make_NEIR0(cases_df, population_df, w_place, w_date)
        r0_samples, used_brazil = estimate_r0(cases_df, w_place, SAMPLE_SIZE,
                                              MIN_DAYS_r0_ESTIMATE, w_date)
        r0_dist = r0_samples[:, -1]
        w_params = make_param_widgets(NEIR0, t_max = t_max)
        model = SEIRBayes(**w_params, r0_dist=r0_dist)
        model_output = model.sample(SAMPLE_SIZE)
        ei_df = make_EI_df(model, model_output, SAMPLE_SIZE)

        resultado = ei_df.loc[ei_df['day'] > 0].groupby(['day']).agg(f)  # TIRAR PRIMEIRA LINHA
        resultado.index = (pd.date_range(start=w_date, periods=t_max - 1) + pd.DateOffset(1)).strftime(
            '%Y-%m-%d').set_names(['date'])
        resultado.columns = ["_".join(x).lower() for x in resultado.columns.ravel()]
        resultado = resultado.reset_index()

        out = resultado[resultado.columns[resultado.columns.str.contains('infected|date')]].head(15)
        out['state'] = state
        out['train_date'] = w_date

        output = output.append(out)

    rename_cols = {
        'state': 'id',
        'train_date': 'fold',
        'date': 'date_pred',
        'infected_mean': 'pred'
    }
    target = 'num_casos'
    output = output[['state', 'train_date', 'date', 'infected_mean']].rename(columns=rename_cols)

    results_file_name = 'seir__' + target + '.csv'
    bucket_out = 'todospelasaude-lake-deploys'  # already created on S3
    key = 'predictions/states/' + results_file_name

    data_location = 's3://{}/{}'.format(bucket_out, key)
    
    try:
        output = pd.concat([pd.read_csv(data_location, sep=',')[['id', 'fold', 'date_pred', 'pred']], output])
    except FileNotFoundError:
        pass
    finally:
        csv_buffer = StringIO()
        output.to_csv(csv_buffer)
        # print(csv_buffer)
        s3_resource = boto3.resource('s3')
        s3_resource.Object(bucket_out, key).put(Body=csv_buffer.getvalue())


if __name__ == '__main__':
    main()
