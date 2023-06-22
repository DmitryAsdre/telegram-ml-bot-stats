import pandas as pd
from .tg_bot import TGSummaryWriterBase


class TGTableSummaryWriter(TGSummaryWriterBase):
    def __init__(self, path_to_credentials, experiment_name):
        super().__init__(path_to_credentials, experiment_name)

        self.records = pd.DataFrame()
        self.description = None
        self.send_message(self.experiment_name)

    def add_description(self, text_message, **kwargs):
        self.description = f"{text_message}\n"
        for desc_name in kwargs:
            desc = kwargs[desc_name]
            self.description += f"{desc_name} : {desc}\n"

    def add_record(self, **kwargs):
        df_new_record = pd.DataFrame.from_records(kwargs, index=[0])
        self.records = pd.concat([self.records, df_new_record], ignore_index=True)

    def send(self, sort_by=None, ascending=None, round=None):
        self.send_delimiter()
        tmp_records = self.records

        if sort_by is not None and ascending is not None:
            tmp_records = self.records.sort_values(by=sort_by, ascending=ascending)
        elif sort_by is not None:
            tmp_records = self.records.sort_values(by=sort_by)
        
        if round:
            tmp_records_floats = tmp_records.select_dtypes(include=[float])
            tmp_records[tmp_records_floats.columns] = tmp_records_floats.round(round)
        self.send_pandas_df(tmp_records)


        