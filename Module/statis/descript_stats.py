import pandas as pd
import numpy as np
from copy import deepcopy


class make_beutiful_crosstab:
    
    def __init__(
        self, df, col1, col2,
        vertical=True, merge_style="column", round_num=1
    ):
        self.vertical = vertical
        self.round_num = round_num
        self.merge_style = merge_style
        self.cross_df = pd.crosstab(
            df[col1], df[col2], margins=True, margins_name="Total"
        )
        
        
    def run(self):
        
        if self.merge_style == "element":
            result = self.element_style_cross_table()
        else:
            result = self.column_style_cross_table()
            
        return result
        

    # 행의 % 계산
    def make_percentage_df(self):

        if self.vertical:
            standard_array = self.cross_df[["Total"]].values
        else:
            standard_array = self.cross_df.loc["Total"].values

        # 행과 열에 맞는 연산 진행
        ratio_mat = np.round(
            self.cross_df.values/standard_array * 100,
            self.round_num
        )
        # 각 행에 대한 %를 DataFrame으로 생성
        ratio_df = pd.DataFrame(
            data=ratio_mat, index=self.cross_df.index, columns=self.cross_df.columns
        )
        return ratio_df

    
    # column끼리 붙이는 방식
    def column_style_cross_table(self):
        
        percentage_df = self.make_percentage_df()

        # freq table의 column에 (n)을 붙임.
        copy_cross_df = deepcopy(self.cross_df)
        copy_cross_df.columns = copy_cross_df.columns.astype("str") + "(n)"

        # percentage table의 column에 (n)을 붙임.
        percentage_df.columns = percentage_df.columns.astype("str") + "(%)"

        # DataFrame에 병합
        merge_df = pd.merge(copy_cross_df, percentage_df, left_index=True, right_index=True)

        # 컬럼 순서 조정
        column_stack_list = []
        for i in range(len(copy_cross_df.columns)):
            column_stack_list.append(copy_cross_df.columns[i])
            column_stack_list.append(percentage_df.columns[i])

        merge_df = merge_df[column_stack_list]

        return merge_df
    
    
    def element_style_cross_table(self):

        percentage_df = self.make_percentage_df()

        # Percentage를 ()안에 넣는다.
        str_per_array = percentage_df.astype("str").values
        str_per_array = "(" + str_per_array + ")"

        # 빈도와 붙인다.
        result_array = self.cross_df.astype("str").values + str_per_array

        result = pd.DataFrame(
            result_array,
            index=self.cross_df.index,
            columns=self.cross_df.columns
        )
        return result