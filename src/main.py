import pandas as pd
from bs4 import BeautifulSoup


def run():
    df0 = pd.read_json('data/data.json')
    df = df0.copy()
    return df


def pipeline(df):

    # Columns that are dates
    date_cols = ['event_created', 'event_end',
                 'event_published', 'event_start', 'user_created']
    # Columns we want to convert NaNs to 0
    nan_zero = ['has_header', 'delivery_method']
    # Columns we want to convert NaNs to 'None'
    nan_none = ['country', 'venue_address', 'venue_country',
                'venue_latitude', 'venue_longitude', 'venue_name', 'venue_state']
    # Columns we want to get dummies
    dummy_cols = ['country', 'currency']

    # Renaming column to fraud that will indicate if listing is indeed a fraud
    df.replace(['fraudster_event', 'premium', 'spammer_warn', 'fraudster',
                'spammer_limited', 'spammer_noinvite', 'locked', 'tos_lock',
                'tos_warn', 'fraudster_att', 'spammer_web', 'spammer'], [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0], inplace=True)
    df.rename(index=str, columns={"acct_type": "fraud"}, inplace=True)
    # Converting to datetime
    convert_date(df, date_cols)

    # Converting NaNs to 0
    convert_nan(df, nan_zero, 0)
    # Converting NaNs or empty to 'None'
    convert_nan(df, nan_none, 'None')
    # NaN for one col set as another
    nan_replicate_value(df, 'event_published', 'event_created')
    #nan_replicate_value(df, 'sale_duration', 'sale_duration2')

    df = col_transform(df)

    # Parse html format into string for description
    parse_html(df, 'description')
    # Get dummies for categorical values
    df = dummify(df, dummy_cols)
    # Drop columns

    return df


# Converting columns to datetime
def convert_date(df, list_colnames):
    for col in list_colnames:
        df[col] = pd.to_datetime(df[col], unit='s')
        df[col] = df[col].dt.date

# Converting NaNs to 0


def convert_nan(df, list_colnames, replace_with='None'):
    if replace_with == 0:
        for col in list_colnames:
            df[col].fillna(0, inplace=True)
    else:
        for col in list_colnames:
            df[col].fillna(replace_with, inplace=True)
            df[col].replace('', replace_with, inplace=True)


def nan_replicate_value(df, colA, colB):
    # Replace colA with colB
    df[colA].fillna(df[colB], inplace=True)


def parse_html(df, col):
    list_soup = []
    for desc in df[col]:
        soup = BeautifulSoup(desc, 'html.parser')
        list_soup.append(soup.get_text())
    desc = pd.Series(list_soup)
    df[col] = desc.values


def dummify(df, list_colnames):
    df = pd.get_dummies(data=df, columns=list_colnames)
    return df


# INI AND FALIHA WORK


def col_transform(df):
    df['previous_payouts'] = df['previous_payouts'].apply(len)
    df['event_length'] = (df.event_end-df.event_start).astype('timedelta64[D]')
    # df['time_till_payment']=(df.approx_payout_date-df.event_published).astype('timedelta64[D]')

    # current_revenue_sum = 0
    # current_revenue = []
    # for event in df.ticket_types:
    #     for ticket in event:
    #         current_revenue_sum = ticket['quantity_sold']*ticket['cost']

    #     current_revenue.append(current_revenue_sum)
    # cur_se = pd.Series(current_revenue)
    # df['current_revenue'] = cur_se.values

    revenue_sum = 0
    revenue = []
    for event in df.ticket_types:
        for ticket in event:
            revenue_sum = ticket['quantity_total']*ticket['cost']

        revenue.append(revenue_sum)
    se = pd.Series(revenue)
    df['max_revenue'] = se.values

    return df
