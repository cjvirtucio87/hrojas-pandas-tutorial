# pylint: disable=missing-docstring
import os
import tempfile
import unittest
import uuid
import shutil

import numpy as np
import pandas as pd


STATES = ['GA','FL','fl','NY','NJ','TX']
STATUS = [1,2,3]


def _random_vals_date_range(dataset: list, high: int, date_range: pd.DatetimeIndex) -> list:
    return [
        dataset[np.random.randint(low=0,high=high)]
        for i in range(len(date_range))
    ]


def _create_dataset(count=1) -> list:
    output = []

    for _ in range(count):
        date_range = pd.date_range(start='1/1/2009', end='12/31/2012', freq='W-MON')
        output.extend(
            zip(
                _random_vals_date_range(
                    STATUS,
                    len(STATUS),
                    date_range,
                ),
                _random_vals_date_range(
                    STATES,
                    len(STATES),
                    date_range,
                ),
                np.random.randint(low=25,high=1000,size=len(date_range)),
                date_range,
            ),
        )

    return output


class TestDataset(unittest.TestCase):

    def test__create_dataset(self):
        # pylint: disable=no-self-use
        dataset = _create_dataset()
        temp_dir_path = os.path.join(
            tempfile.gettempdir(),
            str(uuid.uuid1()),
        )
        test_excel_filepath = os.path.join(
            temp_dir_path,
            'test.xlsx',
        )
        try:
            os.mkdir(temp_dir_path)
            dataframe = pd.DataFrame(
                data=dataset,
                columns=[
                    'State',
                    'Status',
                    'CustomerCount',
                    'StatusDate',
                ],
            )
            dataframe.to_excel(test_excel_filepath, index=False)
            self.assertTrue(os.path.isfile(test_excel_filepath))
        finally:
            if os.path.exists(temp_dir_path) and os.path.isdir(temp_dir_path):
                shutil.rmtree(temp_dir_path)
