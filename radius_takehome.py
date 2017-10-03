from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
import plotly.plotly as py
from scipy.stats import chi2_contingency


def get_fill_rates(df):
    '''
    Find fill rate for each field

    Args:
        df: pandas dataframe of business data

    Returns:
        None
    '''
    print df.info()


def get_zip_info(df):
    '''
    Find true-valued fill rate for zip code data
    as well as number of unique zip codes

    Args:
        df: pandas dataframe of business data

    Returns:
        None
    '''
    not_null_zips = df[df['zip'].notnull()]['zip']
    correct_zips = filter(lambda x: len(str(x)) == 5, not_null_zips)
    tvfr_zips = len(correct_zips)
    unique_zips = len(set(correct_zips))
    print 'True-valued Fill Rate for Zip Codes: {}'.format(tvfr_zips)
    print 'Unique Correct Zip Codes: {}'.format(unique_zips)


def get_phone_info(df):
    '''
    Find true-valued fill rate for phone numbers
    as well as number of unique phone numbers

    Args:
        df: pandas dataframe of business data

    Returns:
        None
    '''
    punct = '( )-'
    not_null_phones = df[df['phone'].notnull()]['phone']
    not_null_phones = not_null_phones.astype(str)
    not_null_phones = not_null_phones.apply(lambda x: x.translate(None, punct))
    correct_phones = filter(lambda x: len(x) == 10, not_null_phones)
    tvfr_phones = len(correct_phones)
    unique_phones = len(set(correct_phones))
    print 'True-valued Fill Rate for Phone Numbers: {}'.format(tvfr_phones)
    print 'Unique Correct Phone Numbers: {}'.format(unique_phones)


def get_state_info(df):
    '''
    Find true-valued fill rate for state abreviations
    as well as number of unique states as a check

    Args:
        df: pandas dataframe of business data

    Returns:
        state_abreves: list of state abreviations
    '''
    not_null_states = df[df['state'].notnull()]['state']
    state_abrevs = ['AK', 'AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL',
                          'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
                          'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT',
                          'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH',
                          'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                          'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'VI', 'PR']
    correct_states = filter(lambda x: str(x).upper() in state_abrevs, not_null_states)
    tvfr_states = len(correct_states)
    state_check = len(set(correct_states))
    print 'True-valued Fill Rate for States: {}'.format(tvfr_states)
    print 'Number of states/US Territories: {}'.format(state_check)
    return state_abrevs


def get_category_code_info(df):
    '''
    Find true-valued fill rate for NAICS codes by
    checking codes against NAICS offocial codes.
    Also find the number of unique category codes

    Args:
        df: pandas dataframe of business data

    Returns:
        None
    '''
    naics = pd.read_csv('NAICS.csv')
    codes = df['category_code'].dropna()
    string_codes = codes.apply(lambda x: str(x))
    string_codes = string_codes.apply(lambda x: x[:-2])
    naics_codes = naics['NAICS Code'].dropna()

    # NAICS codes are max 6-digit, but those in the
    # dataset are all 8, so I'm padding each code in the
    # official list with zeros and truncating those in the dataset
    # to 6 digits
    adjusted_codes = naics_codes.apply(lambda x: str(x).ljust(6, '0'))
    filtered_codes = filter(lambda x: str(x) in adjusted_codes.values, string_codes)
    tvfr_codes = len(filtered_codes)
    unique_codes = len(set(filtered_codes))
    print 'True-valued Fill Rate for Codes: {}'.format(tvfr_codes)
    print 'Unique Correct Codes: {}'.format(unique_codes)


def get_address_info(df):
    '''
    Find true-valued fill rate for addresses
    and the number of unique addresses

    Args:
        df: pandas dataframe of business data

    Returns:
        None
    '''
    address = df['address'].dropna()
    string_address = address.apply(lambda x: str(x))
    address_components = string_address.apply(lambda x: x.split(' '))
    correct_addresses = filter(lambda x: x[0].isdigit() and len(x) >= 2, address_components)

    string_addresses = []
    for address in correct_addresses:
        string_addy = " ".join(address)
        string_addresses.append(string_addy)

    tvfr_addresses = len(correct_addresses)
    unique_addresses = len(set(string_addresses))
    print 'True-valued Fill Rate for Addresses: {}'.format(tvfr_addresses)
    print 'Unique Correct Addresses: {}'.format(unique_addresses)


def get_city_info(df):
    '''
    Find true-valued fill rate for cities
    and the number of unique cities

    Args:
        df: pandas datafram of business data

    Returns:
        None
    '''
    city = df['city'].dropna()
    correct_cities = filter(lambda x: len(str(x)) > 2, city)
    tvfr_city = len(correct_cities)
    unique_cities = len(set(correct_cities))
    print 'True-valued Fill Rate for Cities: {}'.format(tvfr_city)
    print 'Unique Cities: {}'.format(unique_cities)


def get_headcount_info(df):
    '''
    Find true valued fill rate for headcount
    and unique headcounts

    Args:
        df: pandas dataframe of business data

    Returns:
        None
    '''
    headcount = df['headcount'].dropna()
    string_headcount = headcount.apply(lambda x: str(x))
    unique_counts = string_headcount.unique()
    print 'Headcounts in dataset: {}'.format(unique_counts)
    desired_counts = ['50 to 99', '1 to 4', '5 to 9',
                      '10 to 19', '20 to 49',
                      '100 to 249', '250 to 499',
                      '500 to 999', 'Over 1,000']
    correct_counts = filter(lambda x: x in desired_counts, string_headcount)
    tvfr_counts = len(correct_counts)
    num_unique_counts = len(desired_counts)
    print 'True-valued Fill Rate for Headcounts: {}'.format(tvfr_counts)
    print 'Unique Correct Headcounts: {}'.format(num_unique_counts)


def get_name_info(df):
    '''
    Find the true valued fill rate for business
    names and number of unique names

    Args:
        df: pandas dataframe of business data

    Returns:
        None
    '''
    names = df['name'].dropna()
    unwanted = ['', ' ', '0', 0, 'A', 'Q', 'none', 'null', '']
    correct_names = filter(lambda x: x not in unwanted, names)
    real_names = filter(lambda x: len(x) > 1, correct_names)
    tvfr_names = len(real_names)
    unique_names = len(set(real_names))
    print 'True-valued Fill Rate for Names: {}'.format(tvfr_names)
    print 'Unique Names: {}'.format(unique_names)


def get_revenue_info(df):
    '''
    Find the true valued fill rate for revenue and
    the number of unique revenues

    Args:
        df: pandas datafram of business data

    Returns:
        None
    '''
    revenues = df['revenue'].apply(lambda x: str(x))
    print 'Revenues in dataset: {}'.format(revenues.unique())
    wanted_revs = ['$20 to 50 Million',
                   'Less Than $500,000',
                   '$500,000 to $1 Million',
                   '$2.5 to 5 Million',
                   '$1 to 2.5 Million',
                   '$5 to 10 Million',
                   '$10 to 20 Million',
                   '$50 to 100 Million',
                   '$100 to 500 Million',
                   'Over $1 Billion',
                   'Over $500 Million']
    correct_revs = filter(lambda x: x in wanted_revs, revenues)
    tvfr_revs = len(correct_revs)
    unique_revs = len(wanted_revs)
    print 'True-valued Fill Rate for Revenues: {}'.format(tvfr_revs)
    print 'Unique Correct Revenues: {}'.format(unique_revs)


def get_time_info(df):
    '''
    Find the true valued fill rate for time in business
    and the number of unique times

    Args:
        df: pandas dataframe of business data

    Returns:
        None
    '''
    times = df['time_in_business'].dropna()
    string_times = times.apply(lambda x: str(x))
    print 'Times In Business in dataset: {}'.format(string_times.unique())
    wanted_times = ['10+ years', '6-10 years', '1-2 years',
                    '3-5 years', 'Less than a year']
    correct_times = filter(lambda x: x in wanted_times, string_times)
    tvfr_times = len(correct_times)
    unique_times = len(wanted_times)
    print 'True-valued Fill Rate for Times: {}'.format(tvfr_times)
    print 'Unique Times: {}'.format(unique_times)


def create_map_plot(version='business'):
    '''
    Create plots of businesses according to State and
    of each state's population

    Args:
        df: pandas dataframe of business data or state population data
        version: ('business', 'populations') string flag indicating which plot to make

    Returns:
        None
    '''
    if version == 'business':
        state_counts = pd.get_dummies(df_clean['state']).sum(axis=0)
        df = state_counts
        locations = state_counts.index
        title1 = 'Number Of Businesses'
        title2 = 'Businesses By State'
        filename = 'business_map'
    elif version == 'population':
        state_pops = pd.read_csv('census-state-populations.csv')
        locations = state_pops['state']
        df = state_pops['pop_est_2014']
        title1 = 'Population (Millions)'
        title2 = 'Population By State'
        filename = 'population_map'

    scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

    data = [dict(
            type='choropleth',
            colorscale=scl,
            autocolorscale=False,
            locations=locations,
            z=df,
            locationmode='USA-states',
            text=None,
            marker=dict(
                line=dict(
                    color='rgb(255,255,255)',
                    width=2
                )),
            colorbar=dict(
                title=title1)
            )]

    layout = dict(
        title=title2,
        geo=dict(
            scope='usa',
            projection=dict(type='albers usa'),
            showlakes=True,
            lakecolor='rgb(255, 255, 255)'),
        autosize=False,
        width=750,
        height=750)

    fig = dict(data=data, layout=layout)
    py.iplot(fig, filename=filename)


def create_state_biz_hist(df_clean):
    '''
    Create histogram of businesses by state

    Args:
        df_clean: pandas dataframe of clean business data

    Returns:
        None
    '''
    state_counts = pd.get_dummies(df_clean['state']).sum(axis=0)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
    state_counts.plot(kind='bar')
    ax.set_ylabel('Counts')
    ax.set_title('Number Businesses Per State')
    ax.set_xlabel('State/Terriroty')
    plt.xticks(rotation=50)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('Businesses_per_state')


def create_ca_biz_plot(df_clean):
    '''
    Create a histogram of Ca businesses by revenue

    Args:
        df_clean: pandas data frame of cleaned business data

    Returns:
        None
    '''
    ca_biz_counts = df_clean[df_clean['state'] == 'CA']['revenue'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
    ca_biz_counts.plot(kind='bar')
    ax.set_ylabel('Counts')
    ax.set_title('Number Of CA Businesses By Revenue')
    ax.set_xlabel('Revenue')
    plt.xticks(rotation=50)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('Ca_biz_by_revenue')


def check_feature_independence(df_clean, features, target):
    '''
    Create a contingency table of features check for independence

    Args:
        df_clean: pandas data frame of cleaned
            business data
        features: list of features to test
        target: target feature

    Returns:
        chi2: test statistic
        p_value: p-value of test of independence, probability of
            observing a sample statistic as extreme as as test statistic
        dof: degrees of freedom

    '''
    print "Checking for feature independece from {}".format(target)
    for feature in features:
        if feature == target:
            continue
        temp_crosstab = pd.crosstab(df_clean[feature], df_clean[target], margins=True)
        chi2, p_value, dof, expected = chi2_contingency(temp_crosstab)
        print "{} : chi2: {} | p-value: {} | dof: {}".format(feature, chi2, p_value, dof)


if __name__ == '__main__':

    df = pd.read_json('data_analysis.json')

# Get fill rates and true-valued fill rates for each data field
    get_fill_rates(df)

    get_zip_info(df)

    get_phone_info(df)

    state_abrevs = get_state_info(df)

    get_category_code_info(df)

    get_address_info(df)

    get_city_info(df)

    get_headcount_info(df)

    get_name_info(df)

    get_revenue_info(df)

    get_time_info(df)

# Check fill rates and true-valued fill rates for
# dataframe of solely California-based data

    ca_df = df[df['state'] == 'CA']
    print 'Find Fill Rates and True-Valued Fill Rates For California'

    get_fill_rates(ca_df)

    get_zip_info(ca_df)

    get_phone_info(ca_df)

    get_state_info(ca_df)

    get_category_code_info(ca_df)

    get_address_info(ca_df)

    get_city_info(ca_df)

    get_headcount_info(ca_df)

    get_name_info(ca_df)

    get_revenue_info(ca_df)

    get_time_info(ca_df)

# Clean dataframe of unwanted/incorrect values and
# add feature of number of missing entries per business
    unwanted = [' ', 'VI', '0', 'null', '', 'none', None, 0]
    df_clean = df.replace(unwanted, 'N/A')
    df_clean['missing'] = (df_clean == 'N/A').sum(axis=1)
    clean_ca_df = df_clean[df_clean['state'] == 'CA']

# Create clean dataframe of California data
    clean_ca_df['missing'] = df_clean[df_clean['state'] == 'CA']['missing']

# Check for feature independence
    features_1 = ['city',
                  'headcount',
                  'revenue',
                  'state',
                  'time_in_business',
                  'zip']
    targets = features_1
    for target in targets:
        check_feature_independence(df_clean, features_1, target)

# Find Percentage of Businesses By State
    print 'Percentage of Total Businesses By State'
    for state in state_abrevs:
        biz_pct = (df_clean['state'] == state).sum() / len(df_clean)
        print state, biz_pct

# Find top 50 cities where most small businesses
# are located
    ca_rev_mask = clean_ca_df['revenue'] == 'Less Than $500,000'
    print 'Top 50 Cities in Ca By Number Of Small Businesses'
    print clean_ca_df[ca_rev_mask]['city'].value_counts()[:50]

# Check percentage of missing data in California
# small businesses ( revenue <$500,000)
    ca_missing = clean_ca_df['missing'] == 1
    missing_num = ((ca_rev_mask) & (ca_missing)).sum()
    total_num = clean_ca_df[ca_rev_mask].size
    pct_missing = missing_num / total_num
    print 'Percentage Of Missing Ca Small Biz. Data: {}'.format(pct_missing)

# Create histogram of businesses by state
    create_state_biz_hist(df_clean)

# Create business data map plot
    create_map_plot(version='business')

# Create population data map plot
    create_map_plot(version='population')

# Create Ca business by revenue histogram
    create_ca_biz_plot(df_clean)
