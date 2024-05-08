import numpy as np



class seperate_random_balance_dataset:
    
    def __init__(self, key_df, key_list, train_ratio=0.8, valid_ratio=0.1, external_bool=False, external_col=None, external_element=None):
        """
        key_df에 대하여 key_list에 대한 집단별 분포의 비율을 고려한 완전 무작위로 데이터 분할.
        ------------------------------------------------------------------------------
        1) external_bool==True인 경우
        >>> test set은 external_col의 external_element에 해당하는 집단이 Hold-out됨.
        >>> valid set은 train_ratio의 여집합임.
        ------------------------------------------------------------------------------
        2) external_bool==False인 경우
        >>> train set을 먼저 train_ratio만큼 뽑음.
        >>> 나머지에 대하여, valid_ratio만큼 뽑음 -> valid_key_df
        >>> valid_ratio의 나머지 -> train_key_df
        """
        self.key_df = key_df
        self.key_list = key_list
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.external_bool = external_bool
        self.external_col = external_col
        self.external_element = external_element
    
    
    def process(self, seed):

        if self.external_bool:
            train_key_df, valid_key_df, test_key_df = self.external_seperate(seed)
        else:
            train_key_df, valid_key_df, test_key_df = self.internal_seperate(seed)
        return train_key_df, valid_key_df, test_key_df
    
    
    def external_seperate(self, seed):
        external_df = self.key_df[self.key_df[self.external_col]==self.external_element].reset_index(drop=True)
        internal_df = self.key_df[self.key_df[self.external_col]!=self.external_element].reset_index(drop=True)
        
        train_key_df, valid_key_df = seperate_key_dataframe(
            key_df=internal_df, key_list=self.key_list, ratio=self.train_ratio, seed=seed
        ).process()
        return train_key_df, valid_key_df, external_df


    def internal_seperate(self, seed):
        Seperate_df, test_key_df = seperate_key_dataframe(
            key_df=self.key_df, key_list=self.key_list, ratio=self.train_ratio, seed=seed
        ).process()
        valid_key_df, train_key_df = seperate_key_dataframe(
            key_df=Seperate_df, key_list=self.key_list, ratio=self.valid_ratio, seed=seed
        ).process()
        return train_key_df, valid_key_df, test_key_df



class seperate_key_dataframe:
    """
    key_df를 분할한다.
    ===================================================================
    1. 1개 이상의 변수(key_list)에 대하여 ratio만큼 균등하게 분할한다.
    2. seed의 적용을 받으며, seed를 전달 받지 않으면 1234로 seed 고정한다.
    3. 데이터는 seed가 고정된 상태로 무작위로 섞어 분할한다.
    4. 데이터 분할 시, 올림으로 분할해 seperate_df가 rest_df보다 1개 더 많을 수
       있다.
   ===================================================================
   input:
   1. key_df: 분할의 대상이 되는 DataFrame
   2. key_list: 분할의 기준이 되는 column list
   3. ratio: 분할 비율
   4. seed: seed 고정 값
   output:
   1. seperate_df: 분할된 DataFrame
   2. rest_df: 나머지 DataFrame
    """
    def __init__(self, key_df, key_list, ratio, seed=1234):
        
        self.key_df = key_df
        self.key_list = key_list
        self.ratio = ratio
        self.unique_key_array = None
        np.random.seed(seed)
        
        
    def process(self):
        # 1. add seperate key column
        self.add_seperate_key_column()
        # 2. Seperate records
        seperate_df, rest_df = self.seperate_dataFrame()
        # 3. remove key column
        del(seperate_df["key"])
        del(rest_df["key"])

        return seperate_df, rest_df
        
        
    def add_seperate_key_column(self):

        self.key_df["key"] = "|"
        for column in self.key_list:
            self.key_df["key"] = self.key_df["key"] + "|" + self.key_df[column].astype('str')
        self.unique_key_array = np.unique(self.key_df["key"])
        
        
    def seperate_dataFrame(self):

        sep_stack_array = np.array([], dtype=np.int64)
        rest_stack_array = np.array([], dtype=np.int64)

        for key in self.unique_key_array:
            # get target key_df records
            one_key_df = self.key_df[self.key_df["key"] == key]
            # shuffle data
            shuffled_index_array = self.shuffle_index_array(df=one_key_df)
            # seperate index array
            sep_array, rest_array = self.seperate_index_array(index_arr=shuffled_index_array)
            sep_stack_array = np.concatenate((sep_stack_array, sep_array))
            rest_stack_array = np.concatenate((rest_stack_array, rest_array))

        sep_df = self.key_df.loc[sep_stack_array]
        rest_df = self.key_df.loc[rest_stack_array]

        return sep_df, rest_df
        
        
    def shuffle_index_array(self, df):
        index_arr = np.array(df.index)
        np.random.shuffle(index_arr)
        return index_arr
    
    
    def seperate_index_array(self, index_arr):
        # get seperate size
        array_size = len(index_arr)
        sep_size = int(np.ceil(array_size * self.ratio))
        # seperate index_array
        sep_array = index_arr[0:sep_size]
        rest_array = index_arr[sep_size:]
        return sep_array, rest_array