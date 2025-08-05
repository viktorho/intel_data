from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd
from core.base_model import FeaturesFm
class DataTableManager:
    """
    Manage a dynamic set of features as a pandas DataFrame,
    marking specified columns as the primary key (index).
    """
    # map JSON dtype → pandas dtype
    _DTYPE_MAP = {
        "string": "object",
        "float": "float64",
        "int": "int64",
        "date": "datetime64[ns]",
    }

    def __init__(self, feature_specs: FeaturesFm):
        """
        feature_specs: dict with key "features" pointing to a list of dicts,
        each dict having keys: name, dtype, evidence, is_primary_key.
        """
        # build Feature objects
        self.features= feature_specs.features
        # determine primary key columns
        self.pk_fields = [f.name for f in self.features if f.is_primary_key]

        # initialize empty DataFrame with correct columns & dtypes
        cols = [f.name for f in self.features]
        df = pd.DataFrame(columns=cols)
        # apply dtypes
        dtype_dict = {
            f.name: self._DTYPE_MAP.get(f.dtype, "object")
            for f in self.features
        }
        df = df.astype(dtype_dict)
        # if there are primary keys, set them as the index
        if self.pk_fields:
            df.set_index(self.pk_fields, inplace=True, drop=False)
        self.df = df

    def insert_record(self, record: Dict[str, Any]):
        """
        Insert a new row into the table. record should have keys
        matching feature names. After insertion, primary-key uniqueness
        is enforced by pandas (i.e., duplicate index will raise).
        """
        # wrap record in a single-row DataFrame
        row = pd.DataFrame([record])
        # cast to correct types
        row = row.astype({f.name: self._DTYPE_MAP.get(f.dtype, "object")
                          for f in self.features})
        # append to existing df
        self.df = pd.concat([self.df, row], ignore_index=True)
        # re-set index if needed
        if self.pk_fields:
            self.df.set_index(self.pk_fields, inplace=True, drop=False)

    def to_sql(self, table_name: str, conn, if_exists: str = "fail"):
        """
        Write the table to a SQL database.
        conn: SQLAlchemy connection/engine
        if_exists: 'fail', 'replace', or 'append'
        """
        self.df.to_sql(table_name, conn, if_exists=if_exists, index=self.pk_fields!=[])

    def schema(self) -> pd.DataFrame:
        """
        Return a DataFrame describing the schema: column name, dtype, evidence, is_primary_key.
        """
        return pd.DataFrame([f.__dict__ for f in self.features])
    
    def fill_column(self, column_name: str, values: List[Any]):
        """
        Fill a column with a list of values in order.
        Raises ValueError if the length doesn't match DataFrame length.
        """
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

        if len(values) != len(self.df):
            raise ValueError(f"Length of values ({len(values)}) does not match number of rows ({len(self.df)}).")

        self.df[column_name] = values
