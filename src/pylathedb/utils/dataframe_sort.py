from nltk import word_tokenize

def sort_dataframe_by_token_length(df):
    local_df=df.copy()
    local_df['name_length']  = 0
    num_number_col = 0
    for col, dt in local_df.dtypes.items():
        if col  == 'name_length':
            continue
        if dt == object:
            local_df["name_length"] += local_df.apply(lambda row: len(word_tokenize(row[col])), axis=1)
        else:
            num_number_col+=1
    local_df['name_length']+=num_number_col
    local_df = local_df.sort_values('name_length',ascending=False).drop('name_length',axis=1)
    return local_df

def sort_dataframe_by_bow_size(df):
    local_df=df.copy()
    local_df["bow"] = local_df.apply(lambda row: set(),axis=1)
    num_number_col = 0
    for col, dt in local_df.dtypes.items():
        if col  == 'bow' or col=='bow_length':
            continue
        if dt == object:
            local_df["bow"] = local_df.apply(lambda row: row['bow'] | set(word_tokenize(row[col])), axis=1)
        else:
            num_number_col+=1
    local_df['bow_lenght']=local_df.apply(lambda row: len(row['bow'])+num_number_col, axis=1)
    local_df = local_df.sort_values('bow_lenght',ascending=False).drop(['bow_lenght','bow'],axis=1)
    return local_df